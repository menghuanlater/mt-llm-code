# -*- coding: UTF-8 -*-
"""
@File   : build_data_from_json_v8.py
@Author : quanlin03
@Date   : 2023/10/13 13:00
@Usage  : TBD
"""
from transformers import AutoTokenizer
from datasets import Dataset
import argparse
import os

IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Arrow Dataset to disk from json, e.g.: python build_data_from_json.py --tokenizer_file_path ./meta-7b --json_file ./data.json --save_to ./saved_hf_dataset --truncation_length 2048")
    parser.add_argument("--tokenizer_file_path", type=str,
                        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/baomengjiao/LLM/llama_pretrain_and_sft/baichuan2")
    parser.add_argument("--is_sp", default=False, action="store_true", help="是否启用子词模式")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--max_length", type=int, default=2048)

    args = parser.parse_args()
    return args


def batched_preprocess_for_sft_char(examples):
    samples = []
    placeholder_ids = tokenizer.convert_tokens_to_ids(["<reserved_107>"] + [c for c in "[对话上文过长截断标记]"]) + [tokenizer.eos_token_id]
    for source in examples["conversation"]:
        conv_ids_list = []  # 将整个对话, 按照用户和坐席逐个进行编码
        for item in source:
            if item["role"] == "用户":
                conv_ids_list.append({
                    "ids": tokenizer.convert_tokens_to_ids(["<reserved_106>"] + [c for c in item["text"]]), "role": "用户"
                })
            else:
                if item["predict_mode"] == "full_ignore":
                    conv_ids_list.append({
                        "ids": tokenizer.convert_tokens_to_ids(["<reserved_107>"] + [c for c in item["text"]]) + [
                            tokenizer.eos_token_id],
                        "role": "坐席", "predict": False, "predict_mode": item["predict_mode"]
                    })
                else:
                    conv_ids_list.append({
                        "ids": tokenizer.convert_tokens_to_ids(["<reserved_107>"] + [c for c in item["text"]]) + [
                            tokenizer.eos_token_id],
                        "role": "坐席", "predict": True, "predict_mode": item["predict_mode"]
                    })
        # 递推式构造样本
        accumulate_length = 0  # 特征的ids长度
        has_predict_flag = False
        start_ptr = 2  # 滑动指针, 表示的是如果构建上下文, 应该从哪句话开始, 初始状态为2
        is_related_truncation = False
        for i in range(len(conv_ids_list)):
            if conv_ids_list[i]["role"] == "用户":
                accumulate_length += len(conv_ids_list[i]["ids"])
                continue
            assert conv_ids_list[i]["role"] == "坐席"
            if accumulate_length + len(conv_ids_list[i]["ids"]) < MAX_LENGTH - 1:
                accumulate_length += len(conv_ids_list[i]["ids"])
                if not has_predict_flag and conv_ids_list[i]["predict"]:
                    has_predict_flag = True
                continue
            else:
                # 先判断是否已经能够预测了, 如果可以的话, 则将前序的组织为一个example
                if has_predict_flag:
                    assert i > start_ptr  # 如果不满足, 说明max_len设置的也太小了
                    if is_related_truncation:
                        input_ids = conv_ids_list[0]["ids"] + placeholder_ids
                        labels = [IGNORE_INDEX] * len(input_ids)
                        for t in conv_ids_list[start_ptr:i]:
                            input_ids.extend(t["ids"])
                            if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                                labels.extend([IGNORE_INDEX] * len(t["ids"]))
                            else:
                                if t["predict_mode"] == "end_ignore":
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                                else:
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:])
                                t["predict"] = False  # 预测结束之后置为False
                    else:
                        input_ids, labels = [], []
                        for t in conv_ids_list[:i]:
                            input_ids.extend(t["ids"])
                            if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                                labels.extend([IGNORE_INDEX] * len(t["ids"]))
                            else:
                                if t["predict_mode"] == "end_ignore":
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                                else:
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:])
                                t["predict"] = False  # 预测结束之后置为False
                    samples.append({"input_ids": input_ids, "labels": labels})
                    has_predict_flag = False  # 记得置空
                # 执行上下文的递归删除流程, 先判断是否修正一下累计长度
                if not is_related_truncation:
                    is_related_truncation = True
                    accumulate_length -= len(conv_ids_list[1]["ids"])
                    accumulate_length += len(placeholder_ids)
                while accumulate_length + len(conv_ids_list[i]["ids"]) > MAX_LENGTH - 1:
                    accumulate_length -= len(conv_ids_list[start_ptr]["ids"]) + len(
                        conv_ids_list[start_ptr + 1]["ids"])
                    start_ptr += 2
                accumulate_length += len(conv_ids_list[i]["ids"])
                if not has_predict_flag and conv_ids_list[i]["predict"]:
                    has_predict_flag = True
        # 循环结束的时候, 需要判断是否有has_predict_flag
        if has_predict_flag:
            if is_related_truncation:
                input_ids = conv_ids_list[0]["ids"] + placeholder_ids
                labels = [IGNORE_INDEX] * len(input_ids)
                for t in conv_ids_list[start_ptr:]:
                    input_ids.extend(t["ids"])
                    if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                        labels.extend([IGNORE_INDEX] * len(t["ids"]))
                    else:
                        if t["predict_mode"] == "end_ignore":
                            labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                        else:
                            labels.extend([IGNORE_INDEX] + t["ids"][1:])
            else:
                input_ids, labels = [], []
                for t in conv_ids_list:
                    input_ids.extend(t["ids"])
                    if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                        labels.extend([IGNORE_INDEX] * len(t["ids"]))
                    else:
                        if t["predict_mode"] == "end_ignore":
                            labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                        else:
                            labels.extend([IGNORE_INDEX] + t["ids"][1:])
            samples.append({"input_ids": input_ids, "labels": labels})

    n_input_ids, n_attention_mask, n_labels = [], [], []
    for sample in samples:
        n_input_ids.append(sample["input_ids"][:])
        n_attention_mask.append([1] * len(sample["input_ids"]))
        n_labels.append(sample["labels"][:])
    tokenized_full_prompt = {
        "input_ids": n_input_ids,
        "attention_mask": n_attention_mask,
        "labels": n_labels
    }
    return tokenized_full_prompt


def batched_preprocess_for_sft_sp(examples):
    samples = []
    placeholder_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<reserved_107>[对话上文过长截断标记]")) + [tokenizer.eos_token_id]
    for source in examples["conversation"]:
        conv_ids_list = []  # 将整个对话, 按照用户和坐席逐个进行编码
        for item in source:
            if item["role"] == "用户":
                conv_ids_list.append({
                    "ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"<reserved_106>{item['text']}")), "role": "用户"
                })
            else:
                if item["predict_mode"] == "full_ignore":
                    conv_ids_list.append({
                        "ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"<reserved_107>{item['text']}")) + [
                            tokenizer.eos_token_id],
                        "role": "坐席", "predict": False, "predict_mode": item["predict_mode"]
                    })
                else:
                    conv_ids_list.append({
                        "ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"<reserved_107>{item['text']}")) + [
                            tokenizer.eos_token_id],
                        "role": "坐席", "predict": True, "predict_mode": item["predict_mode"]
                    })
        # 递推式构造样本
        accumulate_length = 0  # 特征的ids长度
        has_predict_flag = False
        start_ptr = 2  # 滑动指针, 表示的是如果构建上下文, 应该从哪句话开始, 初始状态为2
        is_related_truncation = False
        for i in range(len(conv_ids_list)):
            if conv_ids_list[i]["role"] == "用户":
                accumulate_length += len(conv_ids_list[i]["ids"])
                continue
            assert conv_ids_list[i]["role"] == "坐席"
            if accumulate_length + len(conv_ids_list[i]["ids"]) < MAX_LENGTH - 1:
                accumulate_length += len(conv_ids_list[i]["ids"])
                if not has_predict_flag and conv_ids_list[i]["predict"]:
                    has_predict_flag = True
                continue
            else:
                # 先判断是否已经能够预测了, 如果可以的话, 则将前序的组织为一个example
                if has_predict_flag:
                    assert i > start_ptr  # 如果不满足, 说明max_len设置的也太小了
                    if is_related_truncation:
                        input_ids = conv_ids_list[0]["ids"] + placeholder_ids
                        labels = [IGNORE_INDEX] * len(input_ids)
                        for t in conv_ids_list[start_ptr:i]:
                            input_ids.extend(t["ids"])
                            if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                                labels.extend([IGNORE_INDEX] * len(t["ids"]))
                            else:
                                if t["predict_mode"] == "end_ignore":
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                                else:
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:])
                                t["predict"] = False  # 预测结束之后置为False
                    else:
                        input_ids, labels = [], []
                        for t in conv_ids_list[:i]:
                            input_ids.extend(t["ids"])
                            if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                                labels.extend([IGNORE_INDEX] * len(t["ids"]))
                            else:
                                if t["predict_mode"] == "end_ignore":
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                                else:
                                    labels.extend([IGNORE_INDEX] + t["ids"][1:])
                                t["predict"] = False  # 预测结束之后置为False
                    samples.append({"input_ids": input_ids, "labels": labels})
                    has_predict_flag = False  # 记得置空
                # 执行上下文的递归删除流程, 先判断是否修正一下累计长度
                if not is_related_truncation:
                    is_related_truncation = True
                    accumulate_length -= len(conv_ids_list[1]["ids"])
                    accumulate_length += len(placeholder_ids)
                while accumulate_length + len(conv_ids_list[i]["ids"]) > MAX_LENGTH - 1:
                    accumulate_length -= len(conv_ids_list[start_ptr]["ids"]) + len(
                        conv_ids_list[start_ptr + 1]["ids"])
                    start_ptr += 2
                accumulate_length += len(conv_ids_list[i]["ids"])
                if not has_predict_flag and conv_ids_list[i]["predict"]:
                    has_predict_flag = True
        # 循环结束的时候, 需要判断是否有has_predict_flag
        if has_predict_flag:
            if is_related_truncation:
                input_ids = conv_ids_list[0]["ids"] + placeholder_ids
                labels = [IGNORE_INDEX] * len(input_ids)
                for t in conv_ids_list[start_ptr:]:
                    input_ids.extend(t["ids"])
                    if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                        labels.extend([IGNORE_INDEX] * len(t["ids"]))
                    else:
                        if t["predict_mode"] == "end_ignore":
                            labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                        else:
                            labels.extend([IGNORE_INDEX] + t["ids"][1:])
            else:
                input_ids, labels = [], []
                for t in conv_ids_list:
                    input_ids.extend(t["ids"])
                    if t["role"] == "用户" or (t["role"] == "坐席" and not t["predict"]):
                        labels.extend([IGNORE_INDEX] * len(t["ids"]))
                    else:
                        if t["predict_mode"] == "end_ignore":
                            labels.extend([IGNORE_INDEX] + t["ids"][1:-1] + [IGNORE_INDEX])
                        else:
                            labels.extend([IGNORE_INDEX] + t["ids"][1:])
            samples.append({"input_ids": input_ids, "labels": labels})

    n_input_ids, n_attention_mask, n_labels = [], [], []
    for sample in samples:
        n_input_ids.append(sample["input_ids"][:])
        n_attention_mask.append([1] * len(sample["input_ids"]))
        n_labels.append(sample["labels"][:])
    tokenized_full_prompt = {
        "input_ids": n_input_ids,
        "attention_mask": n_attention_mask,
        "labels": n_labels
    }
    return tokenized_full_prompt


if __name__ == '__main__':

    args = parse_args()
    if not os.path.exists(args.json_file):
        print(f"{args.json_file} not exist.")
        raise FileNotFoundError

    if not os.path.exists(args.tokenizer_file_path):
        print(f"{args.tokenizer_file_path} not exist.")
        raise FileNotFoundError

    print(f'Loading data from {args.json_file}...\n')

    dataset = Dataset.from_json(args.json_file)
    print(f'{dataset.num_rows} rows of data from {args.json_file} loaded')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_file_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    print("tokenizer.bos_token_id: ", tokenizer.bos_token_id)
    print("tokenizer.eos_token_id: ", tokenizer.eos_token_id)

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    MAX_LENGTH = args.max_length

    print(f'Dataset of {dataset.num_rows} data processing...')

    if args.is_sp:
        tokenized_dataset = dataset.map(batched_preprocess_for_sft_sp, batched=True, batch_size=1, num_proc=8,
                                        remove_columns=dataset.column_names)
    else:
        tokenized_dataset = dataset.map(batched_preprocess_for_sft_char, batched=True, batch_size=1, num_proc=8,
                                        remove_columns=dataset.column_names)

    print(f'Dataset saving... Number of rows: {tokenized_dataset.num_rows}')
    tokenized_dataset.save_to_disk(args.save_to)
    print(f'Dataset successfully saved to {args.save_to}')
