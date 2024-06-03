#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author      :   menghuanlater    
@Contact     :   anlin781205936@126.com
@Github      :   https://github.com/menghuanlater
@Create Time :   2021/12/23 17:24

@Note: Pointer_generator baseline
Input: Context [SEP] Answer
Output: Question -> 最大长度51 (开始标记符占据一个位置)

由于涉及到非常多的矩阵乘, 为了避免出错, 这里都是采用torch.einsum实现的, 建议了解一下相关用法, 可以极大降低代码编写难度
'''
import torch
import pickle
from metrics import *
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LRSchedule import WarmUpLinearDecay
from tensorboardX import SummaryWriter
from torch import optim
import datetime
import numpy as np
import argparse


class MyDataset(Dataset):
    def __init__(self, data: list, char_to_id: dict, max_question_len: int, max_paragraph_len: int, is_training: bool):
        super(MyDataset, self).__init__()
        self.data = data
        self.char_to_id = char_to_id
        self.vocab = set(char_to_id.keys())
        self.max_question_len = max_question_len
        self.is_training = is_training
        self.max_paragraph_len = max_paragraph_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = dict(self.data[index])
        # 指针生成网络模型只需要答案的文本即可
        context, question, answer = item["context"], item["question"], item["answer"]["text"]
        encoder_input_ids, decoder_input_ids = [], []
        # 1.1编码context(即paragraph)
        for char in context:
            if char not in self.vocab:
                encoder_input_ids.append(self.char_to_id["[UNK]"])
            else:
                encoder_input_ids.append(self.char_to_id[char])
        encoder_input_ids = encoder_input_ids[:self.max_paragraph_len]  # 进行一次长度截断
        # 1.2拼接[SEP]
        encoder_input_ids.append(self.char_to_id["[SEP]"])
        # 1.3拼接answer
        for char in answer:
            if char not in self.vocab:
                encoder_input_ids.append(self.char_to_id["[UNK]"])
            else:
                encoder_input_ids.append(self.char_to_id[char])

        # 2.1 编码question
        for char in question:
            if char not in self.vocab:
                decoder_input_ids.append(self.char_to_id["[UNK]"])
            else:
                decoder_input_ids.append(self.char_to_id[char])
        # 2.2 判断decoder输入长度
        if len(decoder_input_ids) > self.max_question_len - 1:
            decoder_target = decoder_input_ids[:self.max_question_len]
            decoder_input_ids = [self.char_to_id["[START]"]] + decoder_input_ids[:self.max_question_len - 1]
        else:
            extra = self.max_question_len - 1 - len(decoder_input_ids)
            decoder_target = decoder_input_ids + [self.char_to_id["[END]"]] + [0] * extra
            decoder_input_ids = [self.char_to_id["[START]"]] + decoder_input_ids + [0] * extra
        if self.is_training:
            return {
                "encoder_input_ids": encoder_input_ids, "decoder_input_ids": decoder_input_ids,
                "decoder_target": decoder_target
            }
        else:
            return {
                "encoder_input_ids": encoder_input_ids, "questions": question
            }


def collate_fn(batch):
    # 动态调整当前batch_size的最大输入长度 --> 动态encoder
    max_len = 0
    for item in batch:
        if len(item["encoder_input_ids"]) > max_len:
            max_len = len(item["encoder_input_ids"])
    encoder_input_ids = []
    for item in batch:
        extra = max_len - len(item["encoder_input_ids"])
        encoder_input_ids.append(item["encoder_input_ids"] + [0] * extra)
    if "questions" in batch[0].keys():
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids).long(),
            "questions": [t["questions"] for t in batch]
        }
    else:
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids).long(),
            "decoder_input_ids": torch.tensor([t["decoder_input_ids"] for t in batch]).long(),
            "decoder_target": torch.tensor([t["decoder_target"] for t in batch]).long()
        }


class Encoder(torch.nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, n_layers: int, pretrain_vectors: np.ndarray,
                 dropout_rate: float):
        """
        :param embedding_dimension: 嵌入维度
        :param hidden_dimension: 隐藏层维度
        :param n_layers: Bi-LSTM层数
        :param pretrain_vectors: 预训练嵌入向量
        """
        super(Encoder, self).__init__()
        self.vocab_embedding = torch.nn.Embedding(num_embeddings=len(pretrain_vectors), embedding_dim=embedding_dimension)
        self.vocab_embedding.weight.data = torch.from_numpy(pretrain_vectors).float()

        self.dropout = torch.nn.Dropout(dropout_rate)

        if n_layers > 1:
            self.bi_lstm = torch.nn.LSTM(
                input_size=embedding_dimension, hidden_size=hidden_dimension, dropout=dropout_rate,
                bidirectional=True, num_layers=n_layers, batch_first=True
            )
        else:
            self.bi_lstm = torch.nn.LSTM(
                input_size=embedding_dimension, hidden_size=hidden_dimension,
                bidirectional=True, num_layers=n_layers, batch_first=True
            )

    def forward(self, input_ids):
        """
        :param input_ids: (bsz, seq)
        :return:
        """
        emb = self.dropout(self.vocab_embedding(input_ids))  # (bsz, seq, emb_dim)
        enc_rep, _ = self.bi_lstm(emb)  # (h_0, c_0)置空, 得到(bsz, seq, dim)
        return self.dropout(enc_rep)


class Decoder(torch.nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, n_layers: int, pretrain_vectors: np.ndarray,
                 dropout_rate: float, is_embedding_share: bool, encoder_embeddings):
        super(Decoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)

        if is_embedding_share:
            self.vocab_embedding = encoder_embeddings
        else:
            self.vocab_embedding = torch.nn.Embedding(num_embeddings=len(pretrain_vectors), embedding_dim=embedding_dimension)
            self.vocab_embedding.weight.data = torch.from_numpy(pretrain_vectors).float()

        if n_layers > 1:
            self.lstm = torch.nn.LSTM(
                input_size=embedding_dimension, hidden_size=hidden_dimension, dropout=dropout_rate,
                bidirectional=False, num_layers=n_layers, batch_first=True
            )
        else:
            self.lstm = torch.nn.LSTM(
                input_size=embedding_dimension, hidden_size=hidden_dimension,
                bidirectional=False, num_layers=n_layers, batch_first=True
            )

    def forward(self, input_ids, input_hidden_state=None, input_cell_state=None):
        """
        :param input_ids: (bsz, 1)  --> 逐个传送
        :param input_hidden_state: (n_layer, bsz, dim)
        :param input_cell_state: (n_layer, bsz, dim)
        :return: , h_t, c_t
        """
        emb = self.dropout(self.vocab_embedding(input_ids))  # (bsz, 1, d_emb)
        if input_hidden_state is None and input_cell_state is None:
            _, (h_t, c_t) = self.lstm(emb)
        else:
            _, (h_t, c_t) = self.lstm(emb, (input_hidden_state, input_cell_state))
        return emb[:, 0, :], h_t, c_t


class S2SModel(torch.nn.Module):
    def __init__(self, embedding_dimension, hidden_dimension, n_encoder_layers, n_decoder_layers, pretrain_vectors,
                 dropout_rate, is_embedding_share, lambda_coverage_loss, max_question_len, beam_width, length_penalty,
                 start_token_id, end_token_id):
        super(S2SModel, self).__init__()
        self.encoder = Encoder(
            embedding_dimension=embedding_dimension, dropout_rate=dropout_rate, hidden_dimension=hidden_dimension,
            pretrain_vectors=pretrain_vectors, n_layers=n_encoder_layers
        )
        self.decoder = Decoder(
            embedding_dimension=embedding_dimension, dropout_rate=dropout_rate, hidden_dimension=hidden_dimension,
            pretrain_vectors=pretrain_vectors, n_layers=n_decoder_layers, is_embedding_share=is_embedding_share,
            encoder_embeddings=self.encoder.vocab_embedding
        )
        self.is_embedding_share = is_embedding_share
        self.lambda_coverage_loss = lambda_coverage_loss
        self.max_question_len = max_question_len
        self.hidden_dimension = hidden_dimension
        self.epsilon = 1e-6
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        # 注意力机制参数 + coverage mechanism
        self.attn_linear = torch.nn.Linear(in_features=3 * hidden_dimension, out_features=hidden_dimension, bias=False)
        self.attn_coverage = torch.nn.Parameter(torch.FloatTensor(hidden_dimension))
        self.attn_v = torch.nn.Parameter(torch.FloatTensor(hidden_dimension))
        self.attn_bias = torch.nn.Parameter(torch.FloatTensor(hidden_dimension))

        # vocab decode概率回归参数
        self.output_linear = torch.nn.Linear(in_features=3 * hidden_dimension, out_features=embedding_dimension)
        if is_embedding_share:
            self.generate_bias = torch.nn.Parameter(torch.FloatTensor(len(pretrain_vectors)))
        else:
            self.generate_linear = torch.nn.Linear(in_features=embedding_dimension, out_features=len(pretrain_vectors))

        # 复制机制参数/pointer参数
        self.copy_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * hidden_dimension + embedding_dimension, out_features=1),
            torch.nn.Sigmoid()
        )

    def single_time_decode(self, encoder_rep, encoder_input_ids, decoder_ids, h_t, c_t, sum_attn, calculate_coverage_loss):
        """
        :param decoder_ids: (bsz, 1)
        :param encoder_rep: (bsz, enc_seq_len, 2*dim)
        :param encoder_input_ids: (bsz, enc_seq_len)
        :param h_t: (n_layer, bsz, dim)
        :param c_t: (n_layer, bsz, dim)
        :param sum_attn: (bsz, enc_seq_len)
        :param calculate_coverage_loss: bool
        :return: vocab_prob, h_t', c_t', sum_attn', [coverage_loss_t or None]
        """
        seq_len = encoder_rep.size(1)
        emb_t, h_t, c_t = self.decoder(decoder_ids, h_t, c_t)
        last_layer_h_t = h_t[-1]  # (bsz, hidden_dim)
        # 2.注意力机制, 获得关于enc_rep的summary表示
        attn_vec = self.attn_linear(torch.cat([encoder_rep, last_layer_h_t[:, None, :].repeat(1, seq_len, 1)], dim=-1))
        attn_vec = attn_vec + torch.einsum("bsd,d->bsd", sum_attn[..., None].repeat(1, 1, self.hidden_dimension), self.attn_coverage)  # 融合coverage
        attn_vec = torch.tanh(attn_vec + self.attn_bias)
        attn_score = torch.einsum("bsd,d->bs", attn_vec, self.attn_v)
        attn_prob = torch.softmax(attn_score, dim=-1)
        sum_enc_rep = torch.einsum("bsd,bs->bd", encoder_rep, attn_prob)

        # 3.对vocab做回归
        dec_rep = self.output_linear(torch.cat([sum_enc_rep, last_layer_h_t], dim=-1))  # (bsz, emd_dim)
        if self.is_embedding_share:
            vocab_logits = F.linear(input=dec_rep, weight=self.encoder.vocab_embedding.weight, bias=self.generate_bias)
        else:
            vocab_logits = self.generate_linear(dec_rep)
        vocab_prob = torch.softmax(vocab_logits, dim=-1)  # (bsz, vocab)
        # 4.计算复制机制的结合概率
        combine_prob = self.copy_net(torch.cat([sum_enc_rep, last_layer_h_t, emb_t], dim=-1))  # (bsz, 1)
        # 5.概率结合, 得到当前时刻对每一个字的生成概率
        vocab_prob = combine_prob * vocab_prob
        vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=encoder_input_ids, src=attn_prob * (1 - combine_prob))
        # 6.收尾
        if calculate_coverage_loss:
            coverage_loss_t = torch.sum(torch.where(sum_attn > attn_prob, attn_prob, sum_attn), dim=-1)
            sum_attn += attn_prob
            return vocab_prob, h_t, c_t, sum_attn, coverage_loss_t
        else:
            sum_attn += attn_prob
            return vocab_prob, h_t, c_t, sum_attn

    def forward(self, encoder_input_ids, decoder_input_ids=None, decoder_target=None):
        """
        :param encoder_input_ids: (bsz, enc_seq_len)
        :param decoder_input_ids: (bsz, dec_seq_len)  => 实验中保持dec_seq_len与max_question_len一致
        :param decoder_target: (bsz, dec_seq_len)
        :return:
        """
        encoder_rep = self.encoder(encoder_input_ids)  # (bsz, seq_len, 2*hidden_dim)
        bsz, seq_len = encoder_input_ids.size(0), encoder_input_ids.size(1)
        h_t, c_t = None, None
        if decoder_input_ids is not None and decoder_target is not None:  # 训练模式
            sum_attn = torch.zeros(size=(bsz, seq_len), device=encoder_rep.device).float()
            vocab_prob_list, coverage_loss_list = [], []
            for i in range(self.max_question_len):
                vocab_prob, h_t, c_t, sum_attn, cov_t = self.single_time_decode(
                    encoder_rep=encoder_rep, encoder_input_ids=encoder_input_ids, decoder_ids=decoder_input_ids[:, 0:i + 1],
                    h_t=h_t, c_t=c_t, sum_attn=sum_attn, calculate_coverage_loss=True
                )
                vocab_prob_list.append(vocab_prob[:, None, :])
                coverage_loss_list.append(cov_t[:, None])
            # 计算损失
            predict = torch.cat(vocab_prob_list, dim=1)  # (bsz, dec_len, vocab)
            predict = predict.view(-1, predict.size(-1))  # (bsz * dec_len, vocab)
            decoder_target = decoder_target.view(-1)
            predict = torch.gather(predict, dim=1, index=decoder_target[:, None]).squeeze(dim=-1)
            init_decode_loss = -torch.log(predict + self.epsilon)
            init_coverage_loss = torch.cat(coverage_loss_list, dim=-1).view(-1)
            init_loss = init_decode_loss + self.lambda_coverage_loss * init_coverage_loss
            init_loss *= (decoder_target != 0).float()  # 所有[PAD]的损失不考虑
            loss = torch.sum(init_loss) / torch.nonzero(decoder_target != 0, as_tuple=False).size(0)
            return loss
        else:  # 验证测试模式 --> 使用beam search
            device = encoder_input_ids.device
            decoder_ids = torch.full(size=(bsz * self.beam_width, 1), fill_value=self.start_token_id, dtype=torch.int32).long().to(device)
            encoder_input_ids = encoder_input_ids.repeat(self.beam_width, 1)
            encoder_rep = encoder_rep.repeat(self.beam_width, 1, 1)
            sum_attn = torch.zeros(size=(bsz * self.beam_width, seq_len), device=encoder_rep.device).float()
            dec_topK_log_probs = [0] * (self.beam_width * bsz)  # (bsz*beam)  每个序列的当前log概率和
            dec_topK_sequences = [[] for _ in range(self.beam_width * bsz)] # (bsz*beam, seq_len) 解码id序列
            dec_topK_seq_lens = [1] * (self.beam_width * bsz)  # 解码序列长度 ==> 加上一个偏置项1, 防止进行长度惩罚时出现div 0的情况
            for i in range(1, self.max_question_len + 1):
                vocab_prob, h_t, c_t, sum_attn = self.single_time_decode(
                    encoder_rep=encoder_rep, encoder_input_ids=encoder_input_ids, decoder_ids=decoder_ids,
                    h_t=h_t, c_t=c_t, sum_attn=sum_attn, calculate_coverage_loss=False
                )
                vocab_logp = torch.log(vocab_prob + self.epsilon)  # 取对数， 加eps
                # 进入beam search
                """ step1: 检查是否存在trigram blocking重叠, 只需要检查最后一项和之前项是否存在重叠即可 """
                if i > 4:  # 当序列长度大于等于4时才有意义, 或者当前解码时刻大于4时才有检查的必要
                    for j in range(self.beam_width * bsz):
                        trigram_blocks = []
                        for k in range(3, i):
                            if dec_topK_sequences[j][k-1] == self.end_token_id:
                                break
                            trigram_blocks.append(dec_topK_sequences[j][k-3:k])
                        if len(trigram_blocks) > 1 and trigram_blocks[-1] in trigram_blocks[:-1]:
                            dec_topK_log_probs[j] += -1e9
                """ step2: 为每个样本, 选择topK个序列 ==> 类似于重构dec_topK_sequences """
                for j in range(bsz):
                    topK_vocab_logp = vocab_logp[j::bsz]  # (k, vocab)
                    candidate_list = []
                    """ 容易出错的地方, i=1的时候不需要为每个K生成K个候选,否则beam search将完全沦为greedy search """
                    for k in range(self.beam_width):
                        ind = j + k * bsz
                        if self.end_token_id in dec_topK_sequences[ind]:
                            candidate_list.append({
                                "add_logit": 0, "add_seq_len": 0, "affiliate_k": k, "add_token_id": self.end_token_id,
                                "sort_logits": dec_topK_log_probs[ind] / (dec_topK_seq_lens[ind] ** self.length_penalty)
                            })
                        else:
                            k_logps, k_indices = topK_vocab_logp[k].topk(dim=0, k=self.beam_width)
                            k_logps, k_indices = k_logps.detach().cpu().numpy(), k_indices.detach().cpu().numpy()
                            for l in range(self.beam_width):
                                aff = l if i == 1 else k
                                candidate_list.append({
                                    "add_logit": k_logps[l], "add_seq_len": 1, "affiliate_k": aff, "add_token_id": k_indices[l],
                                    "sort_logits": (dec_topK_log_probs[ind] + k_logps[l]) / ((dec_topK_seq_lens[ind] + 1) ** self.length_penalty)
                                })
                        if i == 1:  ## 当解码第一个词的时候只能考虑一个
                            break
                    candidate_list.sort(key=lambda x: x["sort_logits"], reverse=True)
                    candidate_list = candidate_list[:self.beam_width]
                    """ 序列修正, 更新topK """
                    c_dec_topK_sequences, c_dec_topK_log_probs, c_dec_topK_seq_lens = \
                        deepcopy(dec_topK_sequences), deepcopy(dec_topK_log_probs), deepcopy(dec_topK_seq_lens)
                    for k in range(self.beam_width):
                        ind = bsz * candidate_list[k]["affiliate_k"] + j
                        r_ind = bsz * k + j
                        father_seq, father_logits, father_len = c_dec_topK_sequences[ind], c_dec_topK_log_probs[ind], c_dec_topK_seq_lens[ind]
                        dec_topK_sequences[r_ind] = father_seq + [candidate_list[k]["add_token_id"]]
                        dec_topK_log_probs[r_ind] = father_logits + candidate_list[k]["add_logit"]
                        dec_topK_seq_lens[r_ind] = father_len + candidate_list[k]["add_seq_len"]
                # 更新decoder_ids
                decoder_ids = torch.tensor(dec_topK_sequences).long().to(device)[:, -1:]
            return torch.tensor(dec_topK_sequences[:bsz]).long().to(device)  # (bsz, dec_seq_len)

class InitializeNetWeight(object):
    def __init__(self, init_method, init_range, init_std):
        self.init_method = init_method
        self.init_range = init_range
        self.init_std = init_std

    def _init_weight(self, weight):
        if self.init_method == "normal":
            torch.nn.init.normal_(weight, 0.0, self.init_std)
        elif self.init_method == "uniform":
            torch.nn.init.uniform_(weight, -self.init_range, self.init_range)

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
        elif classname.find("Embedding") != -1:  # 嵌入层采用预训练参数初始化, 这里保持不动
            pass
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                torch.nn.init.normal_(m.weight, 0.0, self.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "attn_v"):
                self._init_weight(m.attn_v)
            if hasattr(m, "attn_coverage"):
                self._init_weight(m.attn_coverage)
            if hasattr(m, "generate_bias"):
                self._init_bias(m.generate_bias)
            if hasattr(m, "attn_bias"):
                self._init_bias(m.attn_bias)

    def init_weights(self, model):
        model.apply(self._init_weights)
        print("random initialize weights succeed.")


class TrainingFramework(object):
    def __init__(self, args):
        self.args = args
        self.device = "cuda:%d" % args.gpu_id

        # 读取数据集和预训练词典
        if args.dataset == "cmrc":
            with open("data/cmrc_train.pkl", "rb") as f:
                train_data = pickle.load(f)
            with open("data/cmrc_dev.pkl", "rb") as f:
                dev_data = pickle.load(f)
            with open("data/cmrc_test.pkl", "rb") as f:
                test_data = pickle.load(f)
        elif args.dataset == "drcd":
            with open("data/drcd_train.pkl", "rb") as f:
                train_data = pickle.load(f)
            with open("data/drcd_dev.pkl", "rb") as f:
                dev_data = pickle.load(f)
            with open("data/drcd_test.pkl", "rb") as f:
                test_data = pickle.load(f)
        else:
            raise Exception("数据集%s不存在, 程序终止" % args.dataset)
        with open("data/vocab.pkl", "rb") as f:
            self.pretrain_vocab = pickle.load(f)

        # 构建dataset 与 dataloader
        train_dataset = MyDataset(data=train_data, char_to_id=self.pretrain_vocab["char_to_id"], is_training=True, max_question_len=args.max_question_len,
                                  max_paragraph_len=args.max_paragraph_len)
        dev_dataset = MyDataset(data=dev_data, char_to_id=self.pretrain_vocab["char_to_id"], is_training=False, max_question_len=args.max_question_len,
                                max_paragraph_len=args.max_paragraph_len)
        test_dataset = MyDataset(data=test_data, char_to_id=self.pretrain_vocab["char_to_id"], is_training=False, max_question_len=args.max_question_len,
                                 max_paragraph_len=args.max_paragraph_len)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        self.dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

        # tensorboard日志路径和模型保存地址
        self.writer = SummaryWriter(
            logdir="tensorboard/{dataset}/pointer_generator".format(dataset=args.dataset)
        )
        self.model_saved_name = "ModelStorage/{dataset}/pointer_generator.pth".format(dataset=args.dataset)

        # 构建模型计算图以及优化器
        self.model = S2SModel(
            embedding_dimension=args.embedding_dimension, hidden_dimension=args.hidden_dimension, n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers, is_embedding_share=args.embedding_share, dropout_rate=args.dropout,
            lambda_coverage_loss=args.lambda_coverage_loss, max_question_len=args.max_question_len, pretrain_vectors=self.pretrain_vocab["vectors"],
            start_token_id=self.pretrain_vocab["char_to_id"]["[START]"], end_token_id=self.pretrain_vocab["char_to_id"]["[END]"],
            beam_width=args.beam_width, length_penalty=args.length_penalty
        )
        init_obj = InitializeNetWeight(init_method=args.init_method, init_range=args.init_range, init_std=args.init_std)
        init_obj.init_weights(self.model)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)
        all_steps = int(args.num_train_epochs * len(train_dataset.data) / args.batch_size)
        self.schedule = WarmUpLinearDecay(
            optimizer=self.optimizer, init_rate=args.lr, warm_up_steps=int(args.warm_up_rate * all_steps),
            decay_steps=int((1.0 - args.warm_up_rate) * all_steps), min_lr_rate=args.min_lr
        )
        self.model.to(self.device)

    def train(self):
        self.model.train()
        steps = 0
        best_metric = 0.0
        for epoch in range(self.args.num_train_epochs):
            for item in self.train_loader:
                self.optimizer.zero_grad()
                encoder_input_ids, decoder_input_ids, decoder_target = item["encoder_input_ids"], item["decoder_input_ids"], item["decoder_target"]
                loss = self.model(
                    encoder_input_ids=encoder_input_ids.to(self.device), decoder_input_ids=decoder_input_ids.to(self.device),
                    decoder_target=decoder_target.to(self.device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                self.writer.add_scalar("train/loss", loss.item(), global_step=steps)
            dev_score, bleu_v, meteor_v = self.eval(is_test=False)
            print("[time: {}] || [epoch: {}] || [valid dev metric : {:.5f}] || [valid bleu score: {:.5f}] || [valid meteor score : {:.5f}]".format(
                datetime.datetime.now(), epoch + 1, dev_score, bleu_v, meteor_v
            ))
            self.writer.add_scalar("valid/dev-%s" % self.args.eval_metric, dev_score, global_step=steps)
            self.writer.add_scalar("valid/bleu", bleu_v, global_step=steps)
            self.writer.add_scalar("valid/meteor", meteor_v, global_step=steps)
            if dev_score > best_metric:
                print("===由于验证指标从%.5f提升到%.5f, 保存模型===" % (best_metric, dev_score))
                best_metric = dev_score
                torch.save(self.model.state_dict(), f=self.model_saved_name)
        print("%s 训练完成, 选择最佳模型保存点执行测试" % str(datetime.datetime.now()))
        self.model.load_state_dict(torch.load(self.model_saved_name, map_location=self.device), strict=True)
        self.model.to(self.device)
        _, bleu_v, meteor_v = self.eval(is_test=True)
        print("%s 测试完成, 测试bleu指标分数为:%.5f, 测试meteor指标分数为:%.5f" % (str(datetime.datetime.now()), bleu_v, meteor_v))

    def eval(self, is_test: bool):
        self.model.eval()
        hypo_list, refer_list = [], []
        loader = self.test_loader if is_test else self.dev_loader
        for item in loader:
            encoder_input_ids, questions = item["encoder_input_ids"], item["questions"]
            refer_list.extend(questions)
            with torch.no_grad():
                decode_seqs = self.model(encoder_input_ids=encoder_input_ids.to(self.device)).cpu().numpy()
            for i in range(len(questions)):
                seq = list(decode_seqs[i])
                s = ""
                for ids in seq:
                    if self.pretrain_vocab["id_to_char"][ids] == "[END]":
                        break
                    if self.pretrain_vocab["id_to_char"][ids] not in ["[START]", "[SEP]", "[PAD]", "[UNK]"]:
                        s += self.pretrain_vocab["id_to_char"][ids]
                hypo_list.append(s)
        self.model.train()
        return self.score(hypo_list, refer_list)

    def score(self, hypo_list: list, refer_list: list):
        dev_v_list, bleu_v_list, meteor_v_list = [], [], []
        assert len(hypo_list) == len(refer_list)
        for i in range(len(hypo_list)):
            _bleu, _meteor = bleu(hypo=list(hypo_list[i]), refer=list(refer_list[i])), meteor(hypo=list(hypo_list[i]), refer=list(refer_list[i]))
            bleu_v_list.append(_bleu)
            meteor_v_list.append(_meteor)
            if self.args.eval_metric == "bleu":
                dev_v_list.append(_bleu)
            elif self.args.eval_metric == "meteor":
                dev_v_list.append(_meteor)
            elif self.args.eval_metric == "rouge_1":
                dev_v_list.append(rouge_1(hypo=list(hypo_list[i]), refer=list(refer_list[i])))
            elif self.args.eval_metric == "rouge_2":
                dev_v_list.append(rouge_2(hypo=list(hypo_list[i]), refer=list(refer_list[i])))
            elif self.args.eval_metric == "rouge_l":
                dev_v_list.append(rouge_l(hypo=list(hypo_list[i]), refer=list(refer_list[i])))
            elif self.args.eval_metric == "bleu_1":
                dev_v_list.append(bleu(hypo=list(hypo_list[i]), refer=list(refer_list[i]), mode=1))
            elif self.args.eval_metric == "bleu_2":
                dev_v_list.append(bleu(hypo=list(hypo_list[i]), refer=list(refer_list[i]), mode=2))
            elif self.args.eval_metric == "bleu_3":
                dev_v_list.append(bleu(hypo=list(hypo_list[i]), refer=list(refer_list[i]), mode=3))
            elif self.args.eval_metric == "bleu_4":
                dev_v_list.append(bleu(hypo=list(hypo_list[i]), refer=list(refer_list[i]), mode=4))
            else:  # 选用默认的rouge_l
                dev_v_list.append(rouge_l(hypo=list(hypo_list[i]), refer=list(refer_list[i])))
        return 100 * np.average(dev_v_list), 100 * np.average(bleu_v_list), 100 * np.average(meteor_v_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于Pointer-generator的问题生成模型")
    # 通用的一些超参数
    parser.add_argument("--dataset", type=str, default="cmrc", help="数据集名称",
                        choices=["cmrc", "drcd"])
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader的异步加载进程数, 设的越大内存占用越多")
    parser.add_argument("--batch_size", type=int, default=32, help="训练的批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率大小")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW使用的权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-9, help="最小的学习率")
    parser.add_argument("--warm_up_rate", type=float, default=0.1, help="热启动过程占整个训练过程的比例")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--gpu_id", type=int, default=0, help="占用哪个GPU显卡")
    parser.add_argument("--num_train_epochs", type=int, default=40, help="最大训练的epoch数")
    parser.add_argument("--init_range", type=float, default=0.1, help="通过U(-init_range, init_range)初始化参数, 当初始化方法为uniform时")
    parser.add_argument("--init_std", type=float, default=0.02, help="通过N(0, init_std)初始化参数, 当初始化方法为normal时")
    parser.add_argument("--init_method", type=str, default="normal", choices=["uniform", "normal"], help="参数初始化方法")
    parser.add_argument("--dropout", type=float, default=0.2, help="随机失活比例")
    parser.add_argument("--eval_metric", type=str, default="rouge_l", help="模型验证时采用哪个评价指标判定当前模型参数是否需要保存",
                        choices=["rouge_l", "rouge_1", "rouge_2", "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor"])

    # 生成式模型专用的超参数
    parser.add_argument("--embedding_share", default=False, action="store_true", help="embedding是否共享 -> encoder输入侧 + decoder输入侧 + decoder输出侧")
    parser.add_argument("--n_encoder_layers", type=int, default=1, help="encoder的Bi-LSTM使用几层")
    parser.add_argument("--n_decoder_layers", type=int, default=1, help="decoder的Uni-LSTM使用几层")
    parser.add_argument("--hidden_dimension", type=int, default=256, help="LSTM隐藏层维度 -> 双向LSTM向量维度为2d")
    parser.add_argument("--embedding_dimension", type=int, default=300, help="预训练词向量/字向量/子词向量的嵌入维度.")
    parser.add_argument("--lambda_coverage_loss", type=float, default=1.0, help="coverage_mechanism的损失加权系数")
    parser.add_argument("--max_question_len", type=int, default=51, help="问题最大长度")
    parser.add_argument("--max_paragraph_len", type=int, default=512, help="段落最大输入长度, 防止溢出, 如果要复现AAAI那篇文章, 需要进行滑窗处理, 比较麻烦")

    # 关于beam_search的一些超参数
    parser.add_argument("--beam_width", type=int, default=5, help="beam search space width")
    parser.add_argument("--length_penalty", type=float, default=0.6, help="length penalty value for beam search")

    args = parser.parse_args()
    print(str(datetime.datetime.now()) + " 创建训练对象")
    obj = TrainingFramework(args)
    print(str(datetime.datetime.now()) + " 训练对象创建完成, 执行训练过程")
    obj.train()
    obj.writer.close()  # 关闭tensorboard日志写对象
    print(str(datetime.datetime.now()) + " 任务完成, 程序结束")

