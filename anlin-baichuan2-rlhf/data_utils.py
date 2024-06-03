# -*- coding: UTF-8 -*-
"""
@File   : data_utils.py
@Author : quanlin03
@Date   : 2023/10/18 14:55
@Usage  : 数据集集成类, 包含偏好数据、PPO的采样数据、PPO训练过程生成的数据

每个原始的数据集文件均为json格式, 考虑到定制化的繁琐, 数据截断等各种预处理逻辑不在RLHF框架内实现, 要求在数据文件生成前即处理好, 因此本文件将不提供
最大长度的截断逻辑, 默认全部长度符合需求, 特别注意, </s>类似的特殊结尾符号也需要一起绑定在数据里
分词逻辑全部采用***子词法***

偏好数据的每一项格式为: {"prompt": string, "better_response": string, "worse_response": string}
PPO数据的每一项格式为: {"prompt": string}

偏好数据采用right-padding的模式, 而ppo数据采用的是left-padding的模式
"""
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import torch
from torch.nn.utils.rnn import pad_sequence


class RewardModelDataset(Dataset):
    def __init__(self, tokenizer, dataset_path):
        parent_dir, filename = os.path.split(dataset_path)
        self.data = load_dataset("json", data_dir=parent_dir, data_files=filename)["train"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt_text, better_response, worse_response = item["prompt"], item["better_response"], item["worse_response"]
        better_input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_text + better_response)) + [self.tokenizer.eos_token_id]
        worse_input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_text + worse_response)) + [self.tokenizer.eos_token_id]
        better_input_ids, worse_input_ids = torch.tensor(better_input_ids).long(), torch.tensor(worse_input_ids).long()
        # 判断数据是否有问题, 需要确保better和worse不一样
        if better_input_ids.size() == worse_input_ids.size() and torch.all(torch.eq(better_input_ids, worse_input_ids)).item():
            raise ValueError(f"偏好数据存在问题, prompt:[{prompt_text}]相同, 但是better和worse的回复内容是一样的[{better_response}], 请检查.")
        return {
            "better_input_ids": better_input_ids, "worse_input_ids": worse_input_ids
        }


# 动态batch
def reward_model_dataset_collate_fn(samples):
    input_ids = [sample["better_input_ids"] for sample in samples] + [
        sample["worse_input_ids"] for sample in samples
    ]  # (2*B, L)
    attention_mask = [
        input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
    ]  # (2*B, L)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    better_input_ids, worse_input_ids = torch.chunk(input_ids, chunks=2, dim=0)
    better_attention_mask, worse_attention_mask = torch.chunk(attention_mask, chunks=2, dim=0)

    return {
        "better_input_ids": better_input_ids, "worse_input_ids": worse_input_ids,
        "better_attention_mask": better_attention_mask, "worse_attention_mask": worse_attention_mask
    }


class PPOModelDataset(Dataset):
    def __init__(self, tokenizer, dataset_path):
        parent_dir, filename = os.path.split(dataset_path)
        self.data = load_dataset("json", data_dir=parent_dir, data_files=filename)["train"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt_text = item["prompt"]
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_text))
        return {
            "input_ids": torch.tensor(input_ids).long()
        }


# 动态batch
def ppo_model_dataset_collate_fn(samples):
    input_ids = [sample["input_ids"] for sample in samples]
    attention_mask = [
        input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
    ]
    input_ids = pad_sequence([item.flip(0) for item in input_ids], batch_first=True, padding_value=0).flip(1)
    attention_mask = pad_sequence([item.flip(0) for item in attention_mask], batch_first=True, padding_value=0).flip(1)

    return {
        "input_ids": input_ids, "attention_mask": attention_mask
    }


if __name__ == "__main__":
    pass



