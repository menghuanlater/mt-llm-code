# -*- coding: UTF-8 -*-
"""
@File   : train_reward_model.py
@Author : quanlin03
@Date   : 2023/10/18 14:56
@Usage  : 训练Reward Model
"""
import argparse
import deepspeed
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, PreTrainedModel
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.deepspeed import HfDeepSpeedConfig
from model_utils import BaichuanModelForScore
from train_utils import *
from data_utils import RewardModelDataset, reward_model_dataset_collate_fn
from transformers import get_scheduler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Baichuan2-7b 奖励模型训练脚本")
    # 模型层
    parser.add_argument("--model_path", type=str, required=True,
                        help="SFT模型文件的地址, 用于初始化奖励模型和读取分词器, 必填")

    # 数据层
    parser.add_argument("--train_dataset_path", type=str, required=True, help="偏好数据训练json文件地址, 必填")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="偏好数据验证json文件地址, 必填")

    # 训练层
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练epoch数, 可选")
    parser.add_argument("--per_device_batch_size", type=int, default=2, help="单卡batch, 设置训练验证一致, 可选")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累计步数, 可选")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",
                        help="时间换空间的梯度检查点开关, 可选")
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率, 可选")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="学习率机制, 可选")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="热启动步数, 可选")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="衰减系数, 可选")
    parser.add_argument("--seed", type=int, default=6666, help="随机种子, 可选")
    parser.add_argument("--fp16", action="store_true", default=False, help="启动fp16训练, 可选")
    parser.add_argument("--bf16", action="store_true", default=False, help="启动bf16训练, 可选")

    # 输出层
    parser.add_argument("--output_dir", type=str, required=True, help="训练模型保存位置, 必填")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "step"],
                        help="模型保存策略, 可选")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="多少步save一次模型, 当保存策略为epoch时, 此参数无意义, 可选")

    # DeepSpeed层
    parser.add_argument("--local_rank", type=int, default=-1, help="进程指示号, 系统自动分配")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], help="Zero优化器模式, 可选")
    parser.add_argument('--offload', action="store_true", default=False, help="是否启动CPU做临时存储, 可选")

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


class RewardModelTrainer(object):
    def __init__(self):
        # 加载模型
        if args.zero_stage == 3:
            self.dstchf = HfDeepSpeedConfig(ds_train_config)
        self.model = BaichuanModelForScore.from_pretrained(args.model_path)
        dist.barrier()  # 进程同步

        # 加载tokenizer和数据集
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/实验探索/生成式对话管理模型/baichuan2/pretrained_models/baichuan2-7b",
            use_fast=False, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.unk_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2

        self.reg_beta = 0.001 * 0.5

        train_dataset = RewardModelDataset(tokenizer=self.tokenizer, dataset_path=args.train_dataset_path)
        eval_dataset = RewardModelDataset(tokenizer=self.tokenizer, dataset_path=args.eval_dataset_path)
        self.train_dataloader = DataLoader(
            dataset=train_dataset, collate_fn=reward_model_dataset_collate_fn,
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=args.per_device_batch_size
        )
        self.eval_dataloader = DataLoader(
            dataset=eval_dataset, collate_fn=reward_model_dataset_collate_fn,
            sampler=DistributedSampler(eval_dataset, shuffle=False),
            batch_size=args.per_device_batch_size
        )

        dist.barrier()  # 进程同步

        # 初始化deepspeed执行引擎相关, 包括优化器等
        args.num_update_steps_per_epoch = (len(self.train_dataloader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        args.total_training_steps = args.num_train_epochs * args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            args.weight_decay,
        )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
        )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.total_training_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=args,
            config=ds_train_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True
        )

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        dist.barrier()  # 进程同步

    def train(self):
        print_rank_0("***** Running reward model training *****", args.global_rank)

        progress_bar = tqdm(
            total=args.num_train_epochs * len(self.train_dataloader),
            desc=f'Training 1/{args.num_train_epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process()
        )

        # 先进行一波eval, 训练开始前
        print_rank_0("***** Evaluating at the beginning *****", args.global_rank)
        print_rank_0("~~~~~ 偏好验证集上模型评分与人工标注一致率为%.5f ~~~~~" % self.eval(), args.global_rank)

        # 开始执行训练
        for epoch in range(args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                better_input_ids = batch["better_input_ids"].to(args.device)
                worse_input_ids = batch["worse_input_ids"].to(args.device)
                better_attention_mask = batch["better_attention_mask"].to(args.device)
                worse_attention_mask = batch["worse_attention_mask"].to(args.device)
                loss = self.loss(
                    better_input_ids=better_input_ids, better_attention_mask=better_attention_mask,
                    worse_input_ids=worse_input_ids, worse_attention_mask=worse_attention_mask
                )
                self.model.backward(loss)
                self.model.step()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                global_step = epoch * len(self.train_dataloader) + step + 1
                progress_bar.set_description(
                    f"Training {epoch + 1}/{args.num_train_epochs} epoch (loss {loss:.4f})")
                progress_bar.update()

                if args.save_strategy == "step" and global_step % args.save_interval == 0:
                    self.save(tag=global_step)
            # epoch结束, 执行验证
            print_rank_0(f"***** Evaluating at epoch {epoch + 1}/{args.num_train_epochs} *****", args.global_rank)
            print_rank_0("~~~~~ 偏好验证集上模型评分与人工标注一致率为%.5f ~~~~~" % self.eval(), args.global_rank)

            if args.save_strategy == "epoch":
                self.save(tag=f"epoch-{epoch + 1}")
            self.model.tput_timer.update_epoch_count()

    def eval(self):
        self.model.eval()
        num_correct_predictions = 0
        num_total_predictions = 0

        for batch in self.eval_dataloader:
            better_input_ids = batch["better_input_ids"].to(args.device)
            worse_input_ids = batch["worse_input_ids"].to(args.device)
            better_attention_mask = batch["better_attention_mask"].to(args.device)
            worse_attention_mask = batch["worse_attention_mask"].to(args.device)
            batch_size = better_input_ids.size(0)

            with torch.no_grad():
                better_end_scores = self.model(
                    better_input_ids, better_attention_mask
                )["end_scores"].squeeze(dim=-1)
                worse_end_scores = self.model(
                    worse_input_ids, worse_attention_mask
                )["end_scores"].squeeze(dim=-1)
            num_correct_predictions += torch.sum(better_end_scores > worse_end_scores)
            num_total_predictions += batch_size

        accuracy = num_correct_predictions / num_total_predictions
        if dist.is_initialized() and dist.get_world_size() > 1:
            accuracy = get_all_reduce_mean(accuracy).item()
        return accuracy

    def loss(self, better_input_ids: torch.LongTensor, better_attention_mask: torch.BoolTensor,
             worse_input_ids: torch.LongTensor, worse_attention_mask: torch.BoolTensor):
        """Loss function for the reward model.
        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.
        Returns:
            dict[str, torch.Tensor]: loss, higher_end_rewards, lower_end_rewards, accuracy
        """
        assert better_input_ids.size(0) == worse_input_ids.size(0)
        scores_dict = self.model(
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0)
        )
        end_scores = scores_dict["end_scores"].squeeze(dim=-1)  # (2 * B,)
        better_end_reward_score, worse_end_reward_score = torch.chunk(end_scores, dim=0, chunks=2)
        reg_l2 = self.reg_beta * (better_end_reward_score ** 2 + worse_end_reward_score ** 2)
        loss = (-F.logsigmoid(better_end_reward_score - worse_end_reward_score) + reg_l2).mean()
        return loss

    def save(self, tag=None):
        """Save model and tokenizer in Hugging Face format."""
        dist.barrier()
        model_to_save: PreTrainedModel = getattr(self.model, "module", self.model)
        if is_main_process():
            model_to_save.config.to_json_file(os.path.join(args.output_dir, "config.json"))
            self.tokenizer.save_pretrained(args.output_dir)
        if tag is not None:
            self.model.save_checkpoint(args.output_dir, tag=tag)
        else:
            self.model.save_checkpoint(args.output_dir)
        dist.barrier()


if __name__ == "__main__":
    args = parse_arguments()
    if args.fp16 == args.bf16:
        raise Exception("bf16和fp16不可同时为True或者False")

    deepspeed.init_distributed()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    args.device = device
    args.global_rank = dist.get_rank()

    seed_everything(args.seed)

    dist.barrier()

    ds_train_config = {
        "train_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps * dist.get_world_size(),
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_param": {
                "device": "none"
            },
            "offload_optimizer": {
                "device": "none"
            },
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": False,
            "max_out_tokens": 2048,
            "inference_tp_size": 1,
            "release_inference_cache": False,
            "pin_parameters": True,
            "tp_gather_partition_size": 8
        }
    }
    if args.fp16:
        ds_train_config["fp16"] = dict()
        ds_train_config["fp16"]["enabled"] = True
    if args.bf16:
        ds_train_config["bf16"] = dict()
        ds_train_config["bf16"]["enabled"] = True
    ds_eval_config = {
        "train_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps * dist.get_world_size(),
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "steps_per_print": 10,
        "zero_optimization": {
            # The evaluation config only works for ZeRO stage 0 and ZeRO stage 3
            "stage": args.zero_stage if args.zero_stage in {0, 3} else 0,
            "stage3_param_persistence_threshold": 1e4,
            "offload_param": {
                "device": "none"
            },
            "memory_efficient_linear": False
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
    if args.fp16:
        ds_eval_config["fp16"] = dict()
        ds_eval_config["fp16"]["enabled"] = True
    if args.bf16:
        ds_eval_config["bf16"] = dict()
        ds_eval_config["bf16"]["enabled"] = True

    trainer = RewardModelTrainer()
    trainer.train()
    print("训练结束")
