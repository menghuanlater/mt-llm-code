# -*- coding: UTF-8 -*-
"""
@File   : train_ppo_model.py
@Author : quanlin03
@Date   : 2023/10/18 15:54
@Usage  : 训练PPO Model, 由于是面向金服场景内对话, 因此本实现的PPO算法会忽略掉ptx那一项, 作为精简
          训练PPO Model, 需要80GBA100显卡
          涉及到的损失和reward score等将记录在output_dir下, 使用tensorboard
          actor、ref、critic、reward均以baichuan2-7B训练得来, tokenizer一致
"""
import argparse
import deepspeed
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from deepspeed.ops.adam import FusedAdam
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import get_scheduler
from train_utils import *
import torch.utils.tensorboard as tensorboard
from modeling_baichuan import BaichuanForCausalLM
from model_utils import BaichuanModelForScore
from data_utils import PPOModelDataset, ppo_model_dataset_collate_fn


def parse_arguments():
    parser = argparse.ArgumentParser(description="Baichuan2-7b PPO模型训练脚本")
    # 模型层
    parser.add_argument("--actor_reference_model_path", type=str, required=True, help="策略模型和参考模型地址, 必填")
    parser.add_argument("--reward_critic_model_path", type=str, required=True, help="奖励模型和批判模型地址, 必填")

    # 数据层
    parser.add_argument("--train_dataset_path", type=str, required=True, help="PPO训练数据地址, 必填")

    # 训练层
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练多少个epoch, 默认1, 可选")
    parser.add_argument("--kl_coeff", type=float, default=0.02, help="参考模型和策略模型之间KL散度的加权值, 可选")
    parser.add_argument("--clip_range_ratio", type=float, default=0.2, help="采用off-policy训练策略时, 控制旧policy数据用于更新当前policy的采样范围, 可选")
    parser.add_argument("--clip_range_score", type=float, default=20, help="将reward_score进行约束, 控制在(-r, r)之间, 可选")
    parser.add_argument("--clip_range_value", type=float, default=5.0, help="值函数取值范围约束在(估计值-r, 估计值+r)之间, 可选")
    parser.add_argument("--update_iters", type=int, default=1, help="旧policy生成的一个batch样本, 迭代更新新policy多少次, 可选")
    parser.add_argument("--per_device_prompt_batch_size", type=int, default=8, help="单卡prompt的batch, 需要大于等于rl-train的, 可选")
    parser.add_argument("--per_device_rl_batch_size", type=int, default=8, help="单卡基于old_policy生成的样本更新rl模型的训练大小, 可选")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计步数, actor-critic相同设置, 可选")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",
                        help="时间换空间的梯度检查点开关, actor-critic相同设置, 可选")
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率, actor-critic相同设置, 可选")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数, actor-critic相同设置, 可选")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="学习率机制, 可选")
    parser.add_argument("--num_warmup_steps", type=int, default=16, help="热启动步数, 可选")
    parser.add_argument("--seed", type=int, default=6666, help="随机种子, 可选")
    parser.add_argument("--fp16", action="store_true", default=False, help="启动fp16训练, 可选")
    parser.add_argument("--bf16", action="store_true", default=False, help="启动bf16训练, 可选")

    # 输出层
    parser.add_argument("--output_dir", type=str, required=True, help="训练的策略模型保存位置, 以及tensorboard日志记录位置, critic不需要保存, 必填")
    parser.add_argument("--save_strategy", type=str, default="step", choices=["epoch", "step"],
                        help="模型保存策略, PPO推荐step, 可选")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="多少步save一次模型, 当保存策略为epoch时, 此参数无意义, 可选")

    # DeepSpeed层
    parser.add_argument("--local_rank", type=int, default=-1, help="进程指示号, 系统自动分配")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], help="Zero优化器模式, 可选")
    parser.add_argument('--offload', action="store_true", default=False, help="是否启动CPU做临时存储, 可选")

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


class PPOModelTrainer(object):
    def __init__(self):
        # PPO的一些参数
        self.gamma = 1.0
        self.lambda_ = 0.95

        # 初始化模型, actor、critic两个模型需要训练, ref、reward两个模型不需要训练
        if args.zero_stage == 3:
            self.dstchf = HfDeepSpeedConfig(ds_train_config)
        self.actor_model = BaichuanForCausalLM.from_pretrained(args.actor_reference_model_path)
        self.ref_model = BaichuanForCausalLM.from_pretrained(args.actor_reference_model_path)  # 不训练
        self.reward_model = BaichuanModelForScore.from_pretrained(args.reward_critic_model_path)  # 不训练
        self.critic_model = BaichuanModelForScore.from_pretrained(args.reward_critic_model_path)

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

        prompt_only_dataset = PPOModelDataset(tokenizer=self.tokenizer, dataset_path=args.train_dataset_path)
        self.prompt_only_dataloader = DataLoader(
            dataset=prompt_only_dataset, collate_fn=ppo_model_dataset_collate_fn,
            sampler=DistributedSampler(prompt_only_dataset, shuffle=True),
            batch_size=args.per_device_prompt_batch_size
        )
        dist.barrier()

        # 初始化deepspeed执行引擎相关, 包括优化器等
        args.num_update_steps_per_epoch = (len(self.prompt_only_dataloader) * args.update_iters * (
            args.per_device_prompt_batch_size // args.per_device_rl_batch_size
        ) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        args.total_training_steps = args.num_train_epochs * args.num_update_steps_per_epoch

        self.actor_model = self._init_train_model_engine(
            model=self.actor_model, weight_decay=args.weight_decay, lr=args.lr
        )
        self.ref_model = self._init_eval_model_engine(model=self.ref_model)
        self.ref_model.eval()
        self.reward_model = self._init_eval_model_engine(model=self.reward_model)
        self.reward_model.eval()
        self.critic_model = self._init_train_model_engine(
            model=self.critic_model, weight_decay=args.weight_decay, lr=args.lr
        )
        if args.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_enable()
            self.critic_model.gradient_checkpointing_enable()

        dist.barrier()  # 进程同步

        # 初始化tensorboard日志
        if is_main_process():
            self.writer = tensorboard.SummaryWriter(args.output_dir)

        # 生成模型配置
        self.generation_config = {
            "temperature": 0.3, "top_k": 5, "top_p": 0.85, "do_sample": True, "num_beams": 1,
            "repetition_penalty": 1.10, "max_new_tokens": 128, "no_repeat_ngram_size": 6
        }

    def set_model_status(self, is_train: bool):
        if is_train:
            self.actor_model.train()
            self.critic_model.train()
        else:
            self.actor_model.eval()
            self.critic_model.eval()

    @staticmethod
    def _init_train_model_engine(model: torch.nn.Module, weight_decay: float, lr: float):
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95))
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.total_training_steps
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_train_config,
            args=args,
            dist_init_required=True
        )
        return engine

    @staticmethod
    def _init_eval_model_engine(model: torch.nn.Module):
        engine, *_ = deepspeed.initialize(
            model=model,
            config=ds_eval_config
        )
        return engine

    def train(self):
        print_rank_0("***** Running PPO model training *****", args.global_rank)

        progress_bar = tqdm(
            total=args.num_train_epochs * len(self.prompt_only_dataloader),
            desc=f'Training 1/{args.num_train_epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process()
        )
        global_step = 0
        for epoch in range(args.num_train_epochs):
            for batch in self.prompt_only_dataloader:
                # 设置eval状态, old_policy去生成trajectory
                self.set_model_status(is_train=False)
                trajectories = self.generate_old_policy_trajectories(batch["input_ids"], batch["attention_mask"])
                self.set_model_status(is_train=True)
                rl_info = None
                for _ in range(args.update_iters):
                    for rl_batch in trajectories:
                        rl_info = self.rl_step(rl_batch)
                        # 记录日志
                        if is_main_process():
                            for key, value in rl_info.items():
                                self.writer.add_scalar("train/" + key, value, global_step=global_step)
                global_step += 1
                progress_bar.set_description(
                    f"Training {epoch + 1}/{args.num_train_epochs} epoch (actor_loss {rl_info['actor_loss']:.4f}, "
                    f"critic_loss {rl_info['critic_loss']:.4f}, reward {rl_info['reward']:.4f})")
                progress_bar.update(1)
                if args.save_strategy == "step" and global_step % args.save_interval == 0:
                    self.save(tag=global_step)
            if args.save_strategy == "epoch":
                self.save(tag=f"epoch-{epoch + 1}")
            self.actor_model.tput_timer.update_epoch_count()
            self.critic_model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def generate_old_policy_trajectories(self, prompt_input_ids: torch.Tensor, prompt_attention_mask: torch.Tensor):
        rl_micro_batches = []
        micro_batch_size = args.per_device_rl_batch_size
        for i in range(0, prompt_input_ids.size(0), micro_batch_size):
            micro_input_ids = prompt_input_ids[i: i + micro_batch_size].to(args.device)
            micro_attention_mask = prompt_attention_mask[i: i + micro_batch_size].to(args.device)
            g_out = self.actor_model.module.generate(
                input_ids=micro_input_ids, attention_mask=micro_attention_mask, synced_gpus=True,
                max_new_tokens=self.generation_config["max_new_tokens"], do_sample=self.generation_config["do_sample"],
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                temperature=self.generation_config["temperature"], top_k=self.generation_config["top_k"],
                top_p=self.generation_config["top_p"], num_beams=self.generation_config["num_beams"],
                repetition_penalty=self.generation_config["repetition_penalty"],
                no_repeat_ngram_size=self.generation_config["no_repeat_ngram_size"]
            )
            dist.barrier()
            attention_mask = torch.logical_and(
                g_out.not_equal(self.tokenizer.pad_token_id),
                g_out.not_equal(self.tokenizer.unk_token_id)
            )  # 因为batch的原因, 所以会有尾部补PAD的操作, UNK顺便也mask以下, 这样方便后续的reward_model计算分数
            logits = self.actor_model(g_out, attention_mask=attention_mask).logits
            ref_logits = self.ref_model(g_out, attention_mask=attention_mask).logits

            end_reward_score = self.reward_model(g_out, attention_mask)["end_scores"].squeeze(dim=-1)  # (B,)
            reward_value = self.critic_model(g_out, attention_mask)["sequence_scores"][:, :-1]  # (B, L - 1)

            log_probs = gather_log_probabilities(logits[:, :-1], g_out[:, 1:])  # 获得每一个生成的token(action)对应的原始解码概率
            ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], g_out[:, 1:])  # 取得对应的reference模型的解码概率
            rl_micro_batches.append({
                "prompt_input_ids": micro_input_ids,
                "log_probs": log_probs, "ref_log_probs": ref_log_probs,
                "end_reward_score": end_reward_score, "values": reward_value,
                "generate_input_ids": g_out, "generate_attention_mask": attention_mask,
            })
        return rl_micro_batches

    @staticmethod
    def actor_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor,
                   seq_mask: torch.BoolTensor):
        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - args.clip_range_ratio,
            1.0 + args.clip_range_ratio,
        )
        return torch.sum(torch.maximum(pg_loss1, pg_loss2) * seq_mask) / seq_mask.sum()

    @staticmethod
    def critic_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, seq_mask: torch.BoolTensor):
        values_clipped = torch.clamp(
            values,
            old_values - args.clip_range_value,
            old_values + args.clip_range_value,
        )
        vf_loss1 = torch.square(values - returns)
        vf_loss2 = torch.square(values_clipped - returns)
        return 0.5 * torch.sum(torch.maximum(vf_loss1, vf_loss2) * seq_mask) / seq_mask.sum()

    def rl_step(self, rl_batch):
        """
        :param rl_batch: contain all tensors
        "prompt_input_ids": (B, L1) -> 用于指示从哪个地方开始执行的生成
        "log_probs": (B, L2 - 1) -> 策略模型输出相关action的策略概率
        "ref_log_probs": ref_log_probs (B, L2 - 1) -> reference模型的概率
        "end_reward_score": reward_score (B,) -> 奖励模型的输出值, 修正结尾, 中间未解码完成默认为0的奖励
        "values": reward_value (B, L2 - 1) -> 值函数的估计值
        "generate_input_ids":(B, L2)  -> 生成的action序列
        "generate_attention_mask": (B, L2) -> 生成的action配套的mask
        :return:
        """
        prompt_input_ids = rl_batch["prompt_input_ids"]
        old_log_probs = rl_batch["log_probs"]
        ref_log_probs = rl_batch["ref_log_probs"]
        end_reward_score = rl_batch["end_reward_score"]
        old_values = rl_batch["values"]
        generate_input_ids = rl_batch["generate_input_ids"]
        generate_attention_mask = rl_batch["generate_attention_mask"]

        start = prompt_input_ids.size(1) - 1
        sequence_mask = generate_attention_mask[:, 1:]

        with torch.no_grad():
            # 计算kl散度, 约束原始reward_scores的范围, 获得修正后的reward_score
            old_rewards = -args.kl_coeff * (old_log_probs - ref_log_probs)
            end_reward_score = torch.clamp(
                end_reward_score, min=-args.clip_range_score, max=args.clip_range_score
            )
            for i in range(prompt_input_ids.size(0)):
                end_index = sequence_mask[i].nonzero()[-1].item()
                old_rewards[i, end_index] += end_reward_score[i]

            # 计算优势函数A的值以及Q函数的值
            last_gae_lambda = 0
            advantages_reversed = []
            length = old_rewards.size(-1)
            old_rewards_ = old_rewards * sequence_mask
            old_values_ = old_values * sequence_mask
            for t in reversed(range(start, length)):
                next_values = old_values_[:, t + 1] if t < length - 1 else 0.0
                delta = old_rewards_[:, t] + self.gamma * next_values - old_values_[:, t]
                last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda
                advantages_reversed.append(last_gae_lambda)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + old_values_[:, start:]
            advantages = advantages.detach()

        # 更新actor
        logits = self.actor_model(generate_input_ids, attention_mask=generate_attention_mask).logits
        log_probs = gather_log_probabilities(logits[:, :-1], generate_input_ids[:, 1:])
        # 计算actor_loss
        actor_loss = self.actor_loss(
            log_probs=log_probs[:, start:], old_log_probs=old_log_probs[:, start:],
            advantages=advantages, seq_mask=sequence_mask[:, start:]
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        # 更新critic
        values = self.critic_model(generate_input_ids, generate_attention_mask)["sequence_scores"][:, :-1]
        critic_loss = self.critic_loss(
            values=values[:, start:], old_values=old_values[:, start:], returns=returns, seq_mask=sequence_mask[:, start:]
        )
        self.critic_model.backward(critic_loss)
        self.critic_model.step()

        end_reward_score = end_reward_score.mean()  # 如果PPO优化的好, 那么平均的end_reward_score会越来越高
        if dist.is_initialized():
            dist.reduce(actor_loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(critic_loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(end_reward_score, dst=0, op=dist.ReduceOp.AVG)

        dist.barrier()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'reward': end_reward_score.item()
        }

    def save(self, tag):
        dist.barrier()
        model_to_save: PreTrainedModel = getattr(self.actor_model, "module", self.actor_model)
        if is_main_process():
            model_to_save.config.to_json_file(os.path.join(args.output_dir, "config.json"))
            self.tokenizer.save_pretrained(args.output_dir)
        if tag is not None:
            self.actor_model.save_checkpoint(args.output_dir, tag=tag)
        else:
            self.actor_model.save_checkpoint(args.output_dir)
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
        "train_batch_size": args.per_device_rl_batch_size * args.gradient_accumulation_steps * dist.get_world_size(),
        "train_micro_batch_size_per_gpu": args.per_device_rl_batch_size,
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
    ds_eval_config = {
        "train_batch_size": args.per_device_rl_batch_size * args.gradient_accumulation_steps * dist.get_world_size(),
        "train_micro_batch_size_per_gpu": args.per_device_rl_batch_size,
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
        ds_train_config["fp16"] = dict()
        ds_eval_config["fp16"] = dict()
        ds_train_config["fp16"]["enabled"] = True
        ds_eval_config["fp16"]["enabled"] = True
    if args.bf16:
        ds_train_config["bf16"] = dict()
        ds_eval_config["bf16"] = dict()
        ds_train_config["bf16"]["enabled"] = True
        ds_eval_config["bf16"]["enabled"] = True
    trainer = PPOModelTrainer()
    trainer.train()


