# -*- coding: UTF-8 -*-
"""
@File   : train_utils.py
@Author : quanlin03
@Date   : 2023/10/19 22:44
@Usage  : 跟训练相关的一些常用函数
"""
import torch
import torch.distributed as dist
import os
import random
import numpy as np


__all__ = [
    'is_main_process',
    'get_all_reduce_mean',
    'get_all_reduce_sum',
    'get_optimizer_grouped_parameters',
    'seed_everything',
    'print_rank_0',
    'gather_log_probabilities'
]


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_parameters(module: torch.nn.Module, weight_decay: float):
    """Get parameter groups with customized weight decay value."""
    no_decay_name_set = {'bias', 'LayerNorm.weight'}

    return [
        {
            'params': [
                param
                for name, param in module.named_parameters()
                if (
                    not any(no_decay_name in name for no_decay_name in no_decay_name_set)
                    and param.requires_grad
                )
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in module.named_parameters()
                if (
                    any(no_decay_name in name for no_decay_name in no_decay_name_set)
                    and param.requires_grad
                )
            ],
            'weight_decay': 0.0,
        },
    ]


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the mean."""
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def get_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the sum."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_log_probabilities(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(dim=-1))
    return log_probs_labels.squeeze(dim=-1)

