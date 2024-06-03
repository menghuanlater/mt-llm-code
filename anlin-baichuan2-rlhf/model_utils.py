# -*- coding: UTF-8 -*-
"""
@File   : model_utils.py
@Author : quanlin03
@Date   : 2023/10/19 14:29
@Usage  : BaichuanModelForScore
"""
from typing import ClassVar
import torch
from modeling_baichuan import BaichuanModel, BaichuanPreTrainedModel, BaichuanConfig, PreTrainedModel


class BaichuanModelForScore(BaichuanPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ['lm_head.weight']  # lm_head不需要, score模式替换为打分

    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.model = BaichuanModel(config)
        self.score_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: torch.nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):
        assert attention_mask is not None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs[0]  # size = (B, L, E)
        scores = self.score_head(hidden_states)  # size = (B, L, 1)

        end_scores = []
        for i in range(input_ids.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_scores.append(scores[i, end_index])
        end_scores = torch.stack(end_scores, dim=0)

        return {
            "sequence_scores": scores.squeeze(dim=-1),  # PPO使用 (B, L)
            "end_scores": end_scores  # RM和计算奖励使用  # (B, 1)
        }



