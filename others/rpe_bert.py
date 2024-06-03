# -*- coding: UTF-8 -*-
"""
@Author: quanlin03
@CreateTime: 2021/7/23
@Usage: 基于相对位置编码预训练的BERT系列模型
@Support: T5-RPE, Shaw-RPE, LFHC-RPE, XL-RPE, GCDF-RPE
=> 所有相对位置编码均不采用register_buffer将gather_matrix存储到GPU中
"""

import torch
import numpy as np
import scipy.stats as st


class T5RelPosEmb(torch.nn.Module):
    def __init__(self, n_heads: int, max_relative_distance: int, max_seq_len: int):
        super(T5RelPosEmb, self).__init__()
        self.n_heads = n_heads
        self.max_relative_distance = max_relative_distance
        self.rel_pos_bias = torch.nn.Parameter(torch.FloatTensor(size=(2 * max_relative_distance + 1, n_heads)))

        # gather_matrix部分的设计
        self.min_seq_len, self.max_seq_len = 2, max_seq_len
        self.gather_matrix = []
        self._construct_gather_matrix()

    def _construct_gather_matrix(self):
        for length in range(self.min_seq_len, self.max_seq_len + 1):
            x = np.zeros(shape=(length, length), dtype=np.int)
            for i in range(length):
                for j in range(length):
                    if i - j < -self.max_relative_distance:  # 距离右边/下文的key过远
                        x[i, j] = 0
                    elif i - j > self.max_relative_distance:  # 距离左边/上文的key过远
                        x[i, j] = 2 * self.max_relative_distance
                    else:
                        x[i, j] = self.max_relative_distance + i - j
            x = np.reshape(x, newshape=(-1,))
            self.gather_matrix.append(torch.from_numpy(x).long())

    def forward(self, input_length: int, device):
        gather_indices = self.gather_matrix[input_length - self.min_seq_len].to(device)
        return self.rel_pos_bias[gather_indices].view(input_length, input_length, 1, self.n_heads)


class ShawRelPosEmb(torch.nn.Module):
    def __init__(self, d_embed: int, n_heads: int, max_k: int, max_seq_len: int):
        super(ShawRelPosEmb, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.max_k = max_k
        self.embedding_matrix = torch.nn.Embedding(num_embeddings=2 * max_k + 1, embedding_dim=d_embed)

        # gather_matrix部分的设计
        self.min_seq_len, self.max_seq_len = 2, max_seq_len
        self.gather_matrix = []
        self._construct_gather_matrix()

        # 快速获得所有的embedding参数
        pos_sequence = torch.arange(2 * max_k + 1).long()
        self.register_buffer("pos_sequence", pos_sequence)

    def _construct_gather_matrix(self):
        for length in range(self.min_seq_len, self.max_seq_len + 1):
            x = np.zeros(shape=(length, length), dtype=np.int)
            for i in range(length):
                for j in range(length):
                    if i - j < -self.max_k:  # 距离右边/下文的key过远
                        x[i, j] = i * (2 * self.max_k + 1)
                    elif i - j > self.max_k:  # 距离左边/上文的key过远
                        x[i, j] = (i + 1) * (2 * self.max_k + 1) - 1
                    else:
                        x[i, j] = i * (2 * self.max_k + 1) + self.max_k + i - j
            x = np.reshape(x, newshape=(-1,))
            self.gather_matrix.append(torch.from_numpy(x).long())

    def forward(self, input_length: int, device):
        gather_indices = self.gather_matrix[input_length - self.min_seq_len].to(device)
        return gather_indices, self.embedding_matrix(self.pos_sequence).view(2 * self.max_k + 1, self.n_heads, self.d_head)


class LFHCRelPosEmb(torch.nn.Module):
    def __init__(self, d_embed: int, n_heads: int, max_k: int, max_seq_len: int, cur_layer: int):
        super(LFHCRelPosEmb, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.max_k = max_k
        self.cur_layer = cur_layer
        self.embedding_matrix = torch.nn.Embedding(num_embeddings=2 * max_k + 1, embedding_dim=d_embed)

        # gather_matrix部分的设计
        self.min_seq_len, self.max_seq_len = 2, max_seq_len
        self.gather_matrix = []
        self._construct_gather_matrix()

        # 快速获得所有的embedding参数
        pos_sequence = torch.arange(2 * max_k + 1).long()
        self.register_buffer("pos_sequence", pos_sequence)

    def _construct_gather_matrix(self):
        max_edge_dis = self.max_k * self.cur_layer
        for length in range(self.min_seq_len, self.max_seq_len + 1):
            x = np.zeros(shape=(length, length), dtype=np.int)
            for i in range(length):
                for j in range(length):
                    if i - j < -max_edge_dis:  # 距离右边/下文的距离过远
                        x[i, j] = i * (2 * self.max_k + 1)
                    elif i - j > max_edge_dis:  # 距离左边/上文的聚类过远
                        x[i, j] = (i + 1) * (2 * self.max_k + 1) - 1
                    elif i - j < 0:
                        dis = j - i
                        for k in range(0, self.max_k):
                            if k * self.cur_layer < dis <= (k + 1) * self.cur_layer:
                                x[i, j] = i * (2 * self.max_k + 1) + (self.max_k - (k + 1))
                                break
                    elif i - j > 0:
                        dis = i - j
                        for k in range(0, self.max_k):
                            if k * self.cur_layer < dis <= (k + 1) * self.cur_layer:
                                x[i, j] = i * (2 * self.max_k + 1) + (self.max_k + (k + 1))
                                break
                    else:
                        x[i, j] = i * (2 * self.max_k + 1) + self.max_k
            x = np.reshape(x, newshape=(-1,))
            self.gather_matrix.append(torch.from_numpy(x).long())

    def forward(self, input_length: int, device):
        gather_indices = self.gather_matrix[input_length - self.min_seq_len].to(device)
        return gather_indices, self.embedding_matrix(self.pos_sequence).view(2 * self.max_k + 1, self.n_heads, self.d_head)


class XLRelPosEmb(torch.nn.Module):
    def __init__(self, d_embed: int, n_heads: int, max_relative_distance: int, max_seq_len: int):
        super(XLRelPosEmb, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.max_relative_distance = max_relative_distance
        self.r_net = torch.nn.Linear(in_features=d_embed, out_features=d_embed, bias=False)

        # gather_matrix部分的设计
        self.min_seq_len, self.max_seq_len = 2, max_seq_len
        self.gather_matrix = []
        self._construct_gather_matrix()

        # 其他
        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_embed, 2.0) / self.d_embed))
        pos_seq = torch.arange(start=-max_relative_distance, end=max_relative_distance + 1, step=1.0)
        sinusoid_inp = torch.ger(pos_seq, inv_freq)
        self.register_buffer("sinusoid_matrix", torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1))

    def _construct_gather_matrix(self):
        for length in range(self.min_seq_len, self.max_seq_len + 1):
            x = np.zeros(shape=(length, length), dtype=int)
            for i in range(length):
                for j in range(length):
                    if i - j < -self.max_relative_distance:  # 距离右边/下文的key过远
                        x[i, j] = i * (2 * self.max_relative_distance + 1)
                    elif i - j > self.max_relative_distance:  # 距离左边/上文的key过远
                        x[i, j] = (i + 1) * (2 * self.max_relative_distance + 1) - 1
                    else:
                        x[i, j] = i * (2 * self.max_relative_distance + 1) + self.max_relative_distance - i + j
            x = np.reshape(x, newshape=(-1,))
            self.gather_matrix.append(torch.from_numpy(x).long())

    def forward(self, input_length: int, device):
        gather_indices = self.gather_matrix[input_length - self.min_seq_len].to(device)
        return gather_indices, self.r_net(self.sinusoid_matrix).view(2 * self.max_relative_distance + 1, self.n_heads, self.d_head)


class GCDFRelPosEmb(torch.nn.Module):
    def __init__(self, d_embed: int, n_heads: int, max_relative_distance: int, max_seq_len: int):
        super(GCDFRelPosEmb, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.max_relative_distance = max_relative_distance
        self.r_net = torch.nn.Linear(in_features=d_embed, out_features=d_embed, bias=False)

        # gather_matrix部分的设计
        self.min_seq_len, self.max_seq_len = 2, max_seq_len
        self.gather_matrix = []
        self._construct_gather_matrix()

        # 其他
        pos_seq = np.arange(start=-max_relative_distance, stop=max_relative_distance + 1, step=1.0)
        x = []
        for i in range(1, d_embed + 1):
            x.append(4 * st.norm.cdf(x=pos_seq, loc=0, scale=d_embed ** (i / d_embed)))
        x = torch.from_numpy(np.array(x).transpose((1, 0))).float()
        self.register_buffer("gaussian_matrix", x)

    def _construct_gather_matrix(self):
        for length in range(self.min_seq_len, self.max_seq_len + 1):
            x = np.zeros(shape=(length, length), dtype=int)
            for i in range(length):
                for j in range(length):
                    if i - j < -self.max_relative_distance:  # 距离右边/下文的key过远
                        x[i, j] = i * (2 * self.max_relative_distance + 1)
                    elif i - j > self.max_relative_distance:  # 距离左边/上文的key过远
                        x[i, j] = (i + 1) * (2 * self.max_relative_distance + 1) - 1
                    else:
                        x[i, j] = i * (2 * self.max_relative_distance + 1) + self.max_relative_distance - i + j
            x = np.reshape(x, newshape=(-1,))
            self.gather_matrix.append(torch.from_numpy(x).long())

    def forward(self, input_length: int, device):
        gather_indices = self.gather_matrix[input_length - self.min_seq_len].to(device)
        return gather_indices, self.r_net(self.gaussian_matrix).view(2 * self.max_relative_distance + 1, self.n_heads, self.d_head)


class PositionwiseFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout_norm: float, layer_norm_epsilon: float):
        super(PositionwiseFFN, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.CoreNet = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_inner),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_norm),
            torch.nn.Linear(d_inner, d_model),
            torch.nn.Dropout(p=dropout_norm)
        )
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inp):
        core_out = self.CoreNet(inp)
        output = self.layer_norm(inp + core_out)
        return output


class BiMultiHeadAttn(torch.nn.Module):
    def __init__(self, n_heads: int, d_model: int, max_seq_len: int, dropout_attn: float, dropout_norm: float,
                 rel_method: str, layer_norm_epsilon: float, **kwargs):
        super(BiMultiHeadAttn, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.rel_method = rel_method
        self.max_seq_len = max_seq_len

        self.query_net = torch.nn.Linear(d_model, d_model)
        self.key_net = torch.nn.Linear(d_model, d_model)
        self.value_net = torch.nn.Linear(d_model, d_model)
        self.o_net = torch.nn.Linear(d_model, d_model)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / (self.d_head ** 0.5)

        self.drop_attn = torch.nn.Dropout(p=dropout_attn)
        self.drop_norm = torch.nn.Dropout(p=dropout_norm)

        # according to rel method
        if self.rel_method == "t5":
            self.rel_pos_encoding = T5RelPosEmb(
                n_heads=n_heads, max_seq_len=max_seq_len, max_relative_distance=kwargs["max_relative_distance"]
            )
        elif self.rel_method == "shaw":
            self.rel_pos_encoding = ShawRelPosEmb(
                d_embed=d_model, n_heads=n_heads, max_seq_len=max_seq_len, max_k=kwargs["max_k"]
            )
        elif self.rel_method == "lfhc":
            self.rel_pos_encoding = LFHCRelPosEmb(
                d_embed=d_model, n_heads=n_heads, max_seq_len=max_seq_len, max_k=kwargs["max_k"], cur_layer=kwargs["cur_layer"]
            )
        elif self.rel_method == "xl":
            self.rel_pos_encoding = XLRelPosEmb(
                d_embed=d_model, n_heads=n_heads, max_seq_len=max_seq_len, max_relative_distance=kwargs["max_relative_distance"]
            )
        elif self.rel_method == "gcdf":
            self.rel_pos_encoding = GCDFRelPosEmb(
                d_embed=d_model, n_heads=n_heads, max_seq_len=max_seq_len, max_relative_distance=kwargs["max_relative_distance"]
            )
        else:
            raise Exception("Unknown relative position encoding mechanism.")

        # bias
        if self.rel_method in ["xl", "gcdf"]:
            self.r_r_bias = torch.nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
            self.r_w_bias = torch.nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))

    def forward(self, inp, attn_mask):
        """
        :param inp: 输入的表征, (seq_len, bsz, dim)
        :param attn_mask: 注意力mask, (seq_len, seq_len, bsz)
        :return: 自注意力建模层之后的序列表征 (seq_len, bsz, dim)
        """
        input_length, bsz = inp.size(0), inp.size(1)

        w_head_q = self.query_net(inp).view(input_length, bsz, self.n_heads, self.d_head)
        w_head_k = self.key_net(inp).view(input_length, bsz, self.n_heads, self.d_head)
        w_head_v = self.value_net(inp).view(input_length, bsz, self.n_heads, self.d_head)

        # according to rel method
        if self.rel_method != "t5":
            indices, rel_pos_key = self.rel_pos_encoding(input_length=input_length, device=inp.device)
            if self.rel_method in ["xl", "gcdf"]:
                matmul_query_key = torch.einsum("ibnd,jbnd->ijbn", w_head_q + self.r_w_bias, w_head_k)
                matmul_query_pos = torch.einsum("ibnd,rnd->irbn", w_head_q + self.r_r_bias, rel_pos_key)
            else:
                matmul_query_key = torch.einsum("ibnd,jbnd->ijbn", w_head_q, w_head_k)
                matmul_query_pos = torch.einsum("ibnd,rnd->irbn", w_head_q, rel_pos_key)
            matmul_query_pos = matmul_query_pos.contiguous()
            matmul_query_pos = matmul_query_pos.view(-1, bsz, self.n_heads)[indices].view(input_length, input_length, bsz,
                                                                                          self.n_heads)
            attn_score = matmul_query_key + matmul_query_pos
            attn_score.mul_(self.scale)
        else:
            rel_pos_bias = self.rel_pos_encoding(input_length=input_length, device=inp.device)
            matmul_query_key = torch.einsum("ibnd,jbnd->ijbn", w_head_q, w_head_k)
            attn_score = matmul_query_key.mul_(self.scale)
            attn_score = attn_score + rel_pos_bias
        # 加入mask
        attn_score = attn_score + (1 - attn_mask[:, :, :, None]) * (-1e30)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.drop_attn(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(input_length, bsz, self.d_model)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop_norm(attn_out)
        output = self.layer_norm(inp + attn_out)

        return output


class IdenticalStackEncoderLayer(torch.nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_inner: int, dropout_attn: float, dropout_norm: float,
                 rel_method: str, max_seq_len: int, layer_norm_epsilon: float, **kwargs):
        super(IdenticalStackEncoderLayer, self).__init__()
        if rel_method in ["xl", "gcdf", "t5"]:
            self.enc_attn = BiMultiHeadAttn(
                d_model=d_model, n_heads=n_heads, dropout_attn=dropout_attn, dropout_norm=dropout_norm,
                rel_method=rel_method, max_seq_len=max_seq_len, layer_norm_epsilon=layer_norm_epsilon,
                max_relative_distance=kwargs["max_relative_distance"]
            )
        elif rel_method in ["shaw"]:
            self.enc_attn = BiMultiHeadAttn(
                d_model=d_model, n_heads=n_heads, dropout_attn=dropout_attn, dropout_norm=dropout_norm,
                rel_method=rel_method, max_seq_len=max_seq_len, layer_norm_epsilon=layer_norm_epsilon,
                max_k=kwargs["max_k"]
            )
        elif rel_method in ["lfhc"]:
            self.enc_attn = BiMultiHeadAttn(
                d_model=d_model, n_heads=n_heads, dropout_attn=dropout_attn, dropout_norm=dropout_norm,
                rel_method=rel_method, max_seq_len=max_seq_len, layer_norm_epsilon=layer_norm_epsilon,
                max_k=kwargs["max_k"], cur_layer=kwargs["cur_layer"]
            )
        else:
            raise Exception("Unknown relative position mechanism.")

        self.ffn_layer = PositionwiseFFN(
            d_model=d_model, d_inner=d_inner, dropout_norm=dropout_norm, layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, inp, attn_mask):
        attn_output = self.enc_attn(inp, attn_mask)
        ffn_output = self.ffn_layer(attn_output)

        return ffn_output


class RPEBertModel(torch.nn.Module):
    """
    只给出最后的rep, 与下游任务相接的参数自己构建即可, 比如pooler或者自回归需要的相关参数层, 由外部自行设置和定义, 这里仅仅保留最根本的参数
    """
    def __init__(self, config: dict, rel_method: str, max_seq_len: int):
        super(RPEBertModel, self).__init__()
        self.config = config
        self.rel_method = rel_method
        self.max_seq_len = max_seq_len

        # 此表嵌入矩阵和段嵌入
        self.token_embeddings = torch.nn.Embedding(embedding_dim=config["hidden_dim"], num_embeddings=config["vocab_size"])
        self.segment_embeddings = torch.nn.Embedding(embedding_dim=config["hidden_dim"], num_embeddings=2)
        self.embedding_layer_norm = torch.nn.LayerNorm(config["hidden_dim"], eps=config["layer_norm_eps"])
        self.embedding_drop = torch.nn.Dropout(p=config["dropout_norm"])
        # 多层模型
        self.layers = torch.nn.ModuleList()

        for i in range(config["n_layers"]):
            if rel_method in ["xl", "t5", "gcdf"]:
                self.layers.append(IdenticalStackEncoderLayer(
                    d_model=config["hidden_dim"], d_inner=config["ffn_size"], dropout_attn=config["dropout_attn"],
                    dropout_norm=config["dropout_norm"], n_heads=config["n_heads"], layer_norm_epsilon=config["layer_norm_eps"],
                    max_seq_len=max_seq_len, rel_method=rel_method, max_relative_distance=config["max_relative_distance"]
                ))
            elif rel_method in ["shaw"]:
                self.layers.append(IdenticalStackEncoderLayer(
                    d_model=config["hidden_dim"], d_inner=config["ffn_size"], dropout_attn=config["dropout_attn"],
                    dropout_norm=config["dropout_norm"], n_heads=config["n_heads"], layer_norm_epsilon=config["layer_norm_eps"],
                    max_seq_len=max_seq_len, rel_method=rel_method, max_k=config["max_k"]
                ))
            elif rel_method in ["lfhc"]:
                self.layers.append(IdenticalStackEncoderLayer(
                    d_model=config["hidden_dim"], d_inner=config["ffn_size"], dropout_attn=config["dropout_attn"],
                    dropout_norm=config["dropout_norm"], n_heads=config["n_heads"], layer_norm_epsilon=config["layer_norm_eps"],
                    max_seq_len=max_seq_len, rel_method=rel_method, cur_layer=i+1, max_k=config["max_k"]
                ))
            else:
                raise Exception("Unknown relative position mechanism")

        # 初始化所有参数
        self.init_weights()

    def _init_weight(self, weight):
        torch.nn.init.normal_(weight, 0.0, self.config["initializer_std"])

    @staticmethod
    def _init_bias(bias):
        torch.nn.init.constant_(bias, 0)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "rel_pos_bias"):
                self._init_weight(m.rel_pos_bias)

    # 初始化模型参数
    def init_weights(self):
        self.token_embeddings.apply(self._init_weights)
        self.segment_embeddings.apply(self._init_weights)
        for layer in self.layers:
            layer.apply(self._init_weights)

    def forward(self, input_ids, input_mask, input_seg=None):
        """
        :param input_ids: 输入文本的词表id序列 (bsz, seq_len)
        :param input_mask: 对应的mask (bsz, seq_len)
        :param input_seg: 分段嵌入输入 (bsz, seq_len), 当为None的时候, 自行构建全0
        :return:
        """
        if input_seg is None:
            input_seg = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)

        # transpose
        input_ids = input_ids.transpose(0, 1).contiguous()
        input_mask = input_mask.transpose(0, 1).contiguous()
        input_seg = input_seg.transpose(0, 1).contiguous()

        token_emb = self.token_embeddings(input_ids)

        seg_emb = self.segment_embeddings(input_seg)
        inp = self.embedding_drop(token_emb + seg_emb)
        inp = self.embedding_layer_norm(inp)

        attn_mask = input_mask[None, :, :].repeat(input_ids.size(0), 1, 1)
        for i in range(self.config["n_layers"]):
            inp = self.layers[i](inp, attn_mask)
        enc_rep = inp.transpose(0, 1).contiguous()

        return enc_rep


if __name__ == "__main__":
    import json

    model = RPEBertModel(json.load(open("configs/gcdf_en_config.json", "r", encoding="UTF-8")), rel_method="gcdf", max_seq_len=256)

    print(model.state_dict().keys())



