# -*- coding: UTF-8 -*-
"""
@File   : train_baichuan2.py
@Author : quanlin03
@Date   : 2023/10/13 13:00
@Usage  : TBD
"""
# import os
# LOCAL_RANK = int(os.environ['LOCAL_RANK'])

import argparse
import random
import numpy as np
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import deepspeed


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def seed_everything(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_class,
                             model_name_or_path,
                             torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    model = model_class.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--model_name_or_path", type=str,
                        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/实验探索/生成式对话管理模型/baichuan2/pretrained_models/baichuan2-7b")
    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--warmup_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_file", type=str, default='./ds_cfg_fp16.json')

    # deepspeed features
    parser.add_argument('--offload', action='store_true')

    parser.add_argument('--zero_stage', type=int, default=3)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    print_rank_0(f'*****{torch.cuda.device_count()}*****', args.global_rank)
    print_rank_0(f'*****{torch.distributed.get_world_size()}*****', args.global_rank)

    seed_everything(args.seed)

    print_rank_0("model_name_or_path : " + args.model_name_or_path, args.global_rank)

    model, tokenizer = load_model_and_tokenizer(AutoModelForCausalLM, args.model_name_or_path,
                                                torch_dtype=torch.bfloat16 if "bf16" in args.deepspeed_file else torch.float16)

    # Prepare the data
    train_dataset = Dataset.load_from_disk(args.train_data)
    # eval_dataset = Dataset.load_from_disk(args.eval_data)
    print_rank_0("***** Data load success! *****", args.global_rank)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_strategy="epoch",
        do_eval=False,
        fp16="fp16" in args.deepspeed_file,
        bf16="bf16" in args.deepspeed_file,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=8,
        warmup_ratio=args.warmup_rate,
        logging_steps=1,
        report_to=["none"],
        deepspeed=args.deepspeed_file
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=2),
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    main()
