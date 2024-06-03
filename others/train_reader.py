"""
@Author: quanlin
@CreateTime: 2021/7/30
@Usage:
1. 1个正样本, 23个负样本, 如果负样本数不满足23, 则补全为PAD的文章进去 => CLS SEP等还是保留
2. 模型文件保存路径 dpr_baseline/reader/retriever_name/dataset/reader_model.pth
3. batch_chunk
4. 验证测试的时候只取ip索引的top100文章
5. 数据集构建需要考虑的滑窗问题
6. 多处相同的answer_span合并问题
7. 标题不参与answer抽取
8. 进行span的归一化聚合 ==> 单样本内的跨样本
9. 检测在top1, top5, top10, top20, top50, top100下的结果
10. 计算top1, top5, top10中含有答案的结果 --> top1其实就是em, 这样做可以丰富评价体系
11. 模型验证选择按照DPR模型论文的方式进行, 从TOP50文章选择得分最佳的作为抽取文章, 再从此文章内选择答案得分最高的作为答案, 从第15个epoch开始进行验证
"""
import collections
import datetime
import json

import torch
import os
import pickle
import numpy as np
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()
from RPEBERT import RPEBertModel
from copy import deepcopy
from torch import optim
from tensorboardX import SummaryWriter
from LRSchedule import WarmUpLinearDecay
from answer_utils import qa_en_normalize_answer, qa_zn_normalize_answer


project_root_path = "/home/ldmc/nfs/quanlin/GraduateDesign"  # 整个项目的目录入口
# project_root_path = "/mnt/nfs-storage/quanlin/GraduateDesign"
max_span_len_map = {
    "natural_questions": 10, "trivia_qa": 20, "squad": 20, "cmrc_drcd": 50
}
pretrain_model_prefix_map = {
    "BERT_Base_EN": {"prefix": "ape_en", "is_rpe": False, "rpe_method": ""},
    "RoBERTa_Base_ZN": {"prefix": "ape_zn", "is_rpe": False, "rpe_method": ""},
    "T5_Base_EN": {"prefix": "rpe_t5_en", "is_rpe": True, "rpe_method": "t5"},
    "T5_Base_ZN": {"prefix": "rpe_t5_zn", "is_rpe": True, "rpe_method": "t5"},
    "Shaw_Base_EN": {"prefix": "rpe_shaw_en", "is_rpe": True, "rpe_method": "shaw"},
    "Shaw_Base_ZN": {"prefix": "rpe_shaw_zn", "is_rpe": True, "rpe_method": "shaw"},
    "LFHC_Base_EN": {"prefix": "rpe_lfhc_en", "is_rpe": True, "rpe_method": "lfhc"},
    "LFHC_Base_ZN": {"prefix": "rpe_lfhc_zn", "is_rpe": True, "rpe_method": "lfhc"},
    "XL_Base_EN": {"prefix": "rpe_xl_en", "is_rpe": True, "rpe_method": "xl"},
    "XL_Base_ZN": {"prefix": "rpe_xl_zn", "is_rpe": True, "rpe_method": "xl"},
    "GCDF_Base_EN": {"prefix": "rpe_gcdf_en", "is_rpe": True, "rpe_method": "gcdf"},
    "GCDF_Base_ZN": {"prefix": "rpe_gcdf_zn", "is_rpe": True, "rpe_method": "gcdf"}
}


class ReaderDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int, max_question_len: int, max_title_len: int,
                 max_answer_len: int, dataset_file_path: str, num_train_negatives: int, is_eval: bool):
        super(ReaderDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.max_title_len = max_title_len
        self.max_answer_len = max_answer_len
        self.num_train_negatives = num_train_negatives
        self.is_eval = is_eval
        with open(dataset_file_path, "rb") as f:
            self.data = pickle.load(f)

        # 当负样本不足时, 自动填充的全0模板
        pad_tokens = ["[CLS]"] + ["[PAD]"] * (self.max_seq_len - 2) + ["[SEP]"]
        self.pad_ids = self.tokenizer.convert_tokens_to_ids(pad_tokens)
        self.pad_mask = [1] * self.max_seq_len
        self.pad_seg = [0] * self.max_seq_len
        self.pad_context_mask = [0] * self.max_seq_len  # 所有负样本根本不需要这个玩意

    def __len__(self):
        return len(self.data)

    def slide_window(self, answer: str, text: str, max_len: int):
        """
        :param answer: 当前的答案
        :param text: 当前的文本 -> 包含答案
        :param max_len: 预留给文本的最大长度
        如果答案真的很长, 预留文本最大长度都没办法覆盖, 则直接把answer分词后截断成20个字符
        :return: text_sequence_tokens, start_label_index, end_label_index
        """
        answer_tokens = self.tokenizer.tokenize(answer)[:self.max_answer_len]
        text_tokens = self.tokenizer.tokenize(text)
        # 记录所有符合条件的pair
        all_match_pairs = []
        for i in range(0, len(text_tokens) - len(answer_tokens) + 1):
            if text_tokens[i:i+len(answer_tokens)] == answer_tokens:
                all_match_pairs.append((i, i+len(answer_tokens) - 1))
        if all_match_pairs[0][1] < max_len:  # 第一个pair的end不超过max_len
            offset = 0
        else:
            offset = all_match_pairs[0][1] - max_len + 1
        text_tokens = text_tokens[offset:offset + max_len]
        start_label_index, end_label_index = [], []
        for s, e in all_match_pairs:
            if e - offset < max_len:
                start_label_index.append(s - offset)
                end_label_index.append(e - offset)
            else:  # 这个pair不满足, 后面的更不会满足, 直接退出
                break
        assert len(start_label_index) > 0
        assert len(end_label_index) > 0
        return text_tokens, start_label_index, end_label_index

    def __getitem__(self, index):
        item = self.data[index]
        """ 
        区分train和eval
        1. eval最简单, 只需要给模型提供input_ids等即可
        2. train考虑滑窗, 提供开始和结束label
        """
        if self.is_eval:
            question, answers, top100_ctxs = item["question"], item["answers"], item["top100_ctxs"]
            question_tokens = self.tokenizer.tokenize(question)[:self.max_question_len]
            input_ids = []
            input_mask = []
            input_seg = []
            context_mask = []
            context_range = []  # 阅读器抽取答案可搜索的span范围
            for c_item in top100_ctxs:
                title, text = c_item["title"], c_item["text"]
                title_tokens = self.tokenizer.tokenize(title)[:self.max_title_len]
                text_tokens = self.tokenizer.tokenize(text)[:(self.max_seq_len - len(question_tokens) - len(title_tokens) - 4)]
                input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + title_tokens + ["[SEP]"] + text_tokens + ["[SEP]"]
                ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                mask = [1] * len(ids)
                seg = [0] * (len(question_tokens) + 2) + [1] * (len(ids) - 2 - len(question_tokens))
                extra = self.max_seq_len - len(ids)
                if extra > 0:
                    ids += [0] * extra
                    mask += [0] * extra
                    seg += [1] * extra
                input_ids.append(ids)
                input_mask.append(mask)
                input_seg.append(seg)
                start, end = len(question_tokens) + len(title_tokens) + 3, len(question_tokens) + len(title_tokens) + 3 + len(text_tokens) - 1
                mask = [0] * start + [1] * (end - start + 1) + [0] * (self.max_seq_len - end - 1)
                context_range.append({"start": start, "end": end})
                context_mask.append(mask)
            return {
                "input_ids": input_ids, "input_mask": input_mask, "input_seg": input_seg, "context_range": context_range,
                "answers": answers, "context_mask": context_mask
            }
        else:
            question, answers, bm25_positive_ctxs, ip_top100_ctxs, answer_map = \
                item["question"], item["answers"], item["bm25_positive_ctxs"], item["ip_top100_ctxs"], item["answer_map"]
            question_tokens = self.tokenizer.tokenize(question)[:self.max_question_len]
            shuffle(answers)
            choose_answer = answers[0]  # 选定answer
            positive_ids, negative_ids = answer_map[choose_answer]["positive_ctxs_ids"], answer_map[choose_answer]["negative_ctxs_ids"]
            shuffle(positive_ids)
            shuffle(negative_ids)
            input_ids = []
            input_mask = []
            input_seg = []
            # context mask在下面实现
            start_label = [0] * self.max_seq_len
            end_label = [0] * self.max_seq_len
            # 首先处理正样本
            choose_positive_ids = positive_ids[0]
            if choose_positive_ids.startswith("bm25-"):
                x = bm25_positive_ctxs[int(choose_positive_ids[5:])]
                title, text = x["title"], x["text"]
            else:
                x = ip_top100_ctxs[int(choose_positive_ids[3:])]
                title, text = x["title"], x["text"]
            title_tokens = self.tokenizer.tokenize(title)[:self.max_title_len]
            remain_max_len = self.max_seq_len - len(question_tokens) - len(title_tokens) - 4
            text_tokens, start_label_index, end_label_index = self.slide_window(choose_answer, text, remain_max_len)
            for index in start_label_index:
                start_label[3 + len(question_tokens) + len(title_tokens) + index] = 1
            for index in end_label_index:
                end_label[3 + len(question_tokens) + len(title_tokens) + index] = 1
            pos_input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + title_tokens + ["[SEP]"] + text_tokens + ["[SEP]"]
            pos_ids = self.tokenizer.convert_tokens_to_ids(pos_input_tokens)
            pos_mask = [1] * len(pos_ids)
            pos_seg = [0] * (len(question_tokens) + 2) + [1] * (len(pos_ids) - 2 - len(question_tokens))
            extra = self.max_seq_len - len(pos_ids)
            if extra > 0:
                pos_ids += [0] * extra
                pos_mask += [0] * extra
                pos_seg += [1] * extra
            input_ids.append(pos_ids)
            input_mask.append(pos_mask)
            input_seg.append(pos_seg)
            start, end = len(question_tokens) + len(title_tokens) + 3, len(question_tokens) + len(title_tokens) + 3 + len(text_tokens) - 1
            context_mask = [0] * start + [1] * (end - start + 1) + [0] * (self.max_seq_len - end - 1)
            # 处理负样本
            for ids in negative_ids[:self.num_train_negatives]:
                title, text = ip_top100_ctxs[ids]["title"], ip_top100_ctxs[ids]["text"]
                title_tokens = self.tokenizer.tokenize(title)[:self.max_title_len]
                text_tokens = self.tokenizer.tokenize(text)[:(self.max_seq_len - len(question_tokens) - len(title_tokens) - 4)]
                input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + title_tokens + ["[SEP]"] + text_tokens + ["[SEP]"]
                ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                mask = [1] * len(ids)
                seg = [0] * (len(question_tokens) + 2) + [1] * (len(ids) - 2 - len(question_tokens))
                extra = self.max_seq_len - len(ids)
                if extra > 0:
                    ids += [0] * extra
                    mask += [0] * extra
                    seg += [1] * extra
                input_ids.append(ids)
                input_mask.append(mask)
                input_seg.append(seg)
            # 查看总样本数是否是 num_train_negatives + 1, 不足的话直接全0加入
            extra = self.num_train_negatives + 1 - len(input_ids)
            while extra > 0:
                input_ids.append(deepcopy(self.pad_ids))
                input_mask.append(deepcopy(self.pad_mask))
                input_seg.append(deepcopy(self.pad_seg))
                extra -= 1
            return {
                "input_ids": input_ids, "input_mask": input_mask, "input_seg": input_seg, "context_mask": context_mask,
                "start_label": start_label, "end_label": end_label
            }


def collate_fn(batch):
    """
    规整化dataset的输出数据, 主要目的是为了防止字符串列表被奇怪的合并
    => 如何区分训练与测试 ==> 检查是否包含start_label这一项, 只要batch的第一条数据包含这一项, 就是训练, 否则就是测试
    """
    input_ids = []
    input_mask = []
    input_seg = []
    context_mask = []

    for item in batch:
        input_ids.append(item["input_ids"])
        input_mask.append(item["input_mask"])
        input_seg.append(item["input_seg"])
        context_mask.append(item["context_mask"])

    if "start_label" in batch[0].keys():  # 训练模式
        start_label, end_label = [], []
        for item in batch:
            start_label.append(item["start_label"])
            end_label.append(item["end_label"])
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_mask": torch.tensor(input_mask).float(),
            "input_seg": torch.tensor(input_seg).long(), "context_mask": torch.tensor(context_mask).float(),
            "start_label": torch.tensor(start_label).long(), "end_label": torch.tensor(end_label).long()
        }
    else:  # 验证测试模式
        context_range, answers = [], []
        for item in batch:
            context_range.append(item["context_range"])
            answers.append(item["answers"])
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_mask": torch.tensor(input_mask).float(),
            "input_seg": torch.tensor(input_seg).long(), "context_mask": torch.tensor(context_mask).float(),
            "context_range": context_range, "answers": answers
        }


class MultiTaskReader(torch.nn.Module):
    def __init__(self, pretrain_model_path: str, is_rpe: bool, rpe_method: str, max_seq_len: int):
        """
        :param pretrain_model_path: 预训练语言模型文件夹
        :param is_rpe: 是否是相对位置编码
        :param max_seq_len: 最大文本长度 ==> 仅仅RPE模型会使用
        :param rpe_method: 相对位置编码方法
        """
        super(MultiTaskReader, self).__init__()
        self.is_rpe = is_rpe
        if is_rpe:
            config = json.load(open(os.path.join(pretrain_model_path, "config.json"), "r", encoding="UTF-8"))
            self.encoder = RPEBertModel(config=config, max_seq_len=max_seq_len, rel_method=rpe_method)
            self.encoder.load_state_dict(torch.load(os.path.join(pretrain_model_path, "pytorch_model.bin"), map_location="cpu"))
        else:
            self.encoder = BertModel.from_pretrained(pretrain_model_path)
        # 线性变换
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=768),
            torch.nn.LayerNorm(768, eps=1e-12)
        )
        # 文章相关性得分层
        self.match_layer = torch.nn.Linear(in_features=768, out_features=1)
        # 答案的开始结束层
        self.start_layer = torch.nn.Linear(in_features=768, out_features=1)
        self.end_layer = torch.nn.Linear(in_features=768, out_features=1)

    def forward(self, input_ids, input_mask, input_seg, context_mask, start_label=None, end_label=None):
        """
        比较复杂的一个forward
        :param input_ids: 输入的文本token序列 (bsz, num_docs, seq_len)
        :param input_mask: 针对pad的遮蔽 (bsz, num_docs, seq_len)
        :param input_seg: 段嵌入 (bsz, num_docs, seq_len)
        :param context_mask: 真正属于passage部分的mask, question和title不直接参与答案抽取的损失计算
                ==> eval阶段, (bsz, num_docs, seq_len), train阶段, (bsz, seq_len)
        :param start_label: 开始标记, 表示哪些位置是答案的开始 (bsz, seq_len)
        :param end_label: 结束标记, 表示哪些位置是答案的结束 (bsz, seq_len)
        """
        # 先铺开ids, mask, seg的前两维
        bsz, num_docs, seq_len = input_ids.size(0), input_ids.size(1), input_ids.size(2)
        input_ids = input_ids.view(-1, seq_len)
        input_mask = input_mask.view(-1, seq_len)
        input_seg = input_seg.view(-1, seq_len)
        if self.is_rpe:
            enc_rep = self.encoder(input_ids, input_mask, input_seg)
        else:
            enc_rep = self.encoder(input_ids, input_mask, input_seg).last_hidden_state
        enc_rep = self.transform(enc_rep).view(bsz, num_docs, seq_len, enc_rep.size(-1))
        match_logits = self.match_layer(enc_rep[:, :, 0, :]).squeeze(dim=-1)  # (bsz, num_docs)
        match_prob = torch.softmax(match_logits, dim=-1)  # (bsz, num_docs)
        ini_start_logits = self.start_layer(enc_rep).squeeze(dim=-1)  # (bsz, num_docs, seq_len)
        ini_end_logits = self.end_layer(enc_rep).squeeze(dim=-1)  # (bsz, num_docs, seq_len)
        if start_label is not None:  # 训练模式
            ini_start_logits, ini_end_logits = ini_start_logits[:, 0, :], ini_end_logits[:, 0, :]  # (bsz, seq_len)
            start_logits = ini_start_logits + (1.0 - context_mask) * (-1e30)
            end_logits = ini_end_logits + (1.0 - context_mask) * (-1e30)
            start_prob = torch.softmax(start_logits, dim=-1)  # (bsz, seq_len)
            end_prob = torch.softmax(end_logits, dim=-1)  # (bsz, seq_len)
            # 1. 直接根据正doc的概率计算-log损失
            pos_passage_match_prob = match_prob[:, 0]
            match_loss = torch.mean(-torch.log(pos_passage_match_prob + 1e-9))

            # 2. 获得所有start标记为true的概率和, 并计算损失 => 只取正样本
            pos_passage_start_prob = torch.sum(start_prob * start_label, dim=-1)
            start_loss = torch.mean(-torch.log(pos_passage_start_prob + 1e-9))

            # 3. 获得所有end标记为true的概率和, 并计算损失 => 只取负样本
            pos_passage_end_prob = torch.sum(end_prob * end_label, dim=-1)
            end_loss = torch.mean(-torch.log(pos_passage_end_prob + 1e-9))
            final_loss = match_loss + start_loss + end_loss
            return final_loss
        else:  # 验证/测试
            start_logits = ini_start_logits + (1.0 - context_mask) * (-1e30)
            end_logits = ini_end_logits + (1.0 - context_mask) * (-1e30)
            start_prob = torch.softmax(start_logits, dim=-1)  # (bsz, num_docs, seq_len)
            end_prob = torch.softmax(end_logits, dim=-1)  # (bsz, num_docs, seq_len)
            return match_prob, start_prob, end_prob


class TrainingFramework(object):
    def __init__(self, args):
        self.args = args
        self.device = "cuda:%d" % args.gpu_id

        prefix = "ape_en"  # 默认值
        is_rpe, rpe_method = False, ""
        for key in pretrain_model_prefix_map.keys():
            if key in args.pretrain_model_path:
                prefix = pretrain_model_prefix_map[key]["prefix"]
                is_rpe = pretrain_model_prefix_map[key]["is_rpe"]
                rpe_method = pretrain_model_prefix_map[key]["rpe_method"]
                break
        self.model = MultiTaskReader(
            pretrain_model_path=os.path.join(project_root_path, args.pretrain_model_path),
            is_rpe=is_rpe, max_seq_len=args.max_seq_len, rpe_method=rpe_method
        )
        tokenizer = BertTokenizer(vocab_file=os.path.join(project_root_path, args.pretrain_model_path, "vocab.txt"))
        self.tokenizer = tokenizer
        train_file_path = os.path.join(project_root_path, "ReproduceDPR/Reader/data",
                                       args.dataset, args.prefix_retriever_name, "train.pkl")
        train_dataset = ReaderDataset(
            dataset_file_path=train_file_path, is_eval=False, max_seq_len=args.max_seq_len, max_title_len=args.max_title_len,
            max_answer_len=args.max_answer_len, max_question_len=args.max_question_len, tokenizer=tokenizer,
            num_train_negatives=args.num_train_negatives
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
        )
        dev_file_path = os.path.join(project_root_path, "ReproduceDPR/Reader/data",
                                     args.dataset, args.prefix_retriever_name, "dev.pkl")
        dev_dataset = ReaderDataset(
            dataset_file_path=dev_file_path, is_eval=True, max_seq_len=args.max_seq_len, max_title_len=args.max_title_len,
            max_answer_len=args.max_answer_len, max_question_len=args.max_question_len, tokenizer=tokenizer,
            num_train_negatives=args.num_train_negatives
        )
        self.dev_loader = DataLoader(
            dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
        )
        if args.not_do_test:
            self.test_loader = None
        else:
            test_file_path = os.path.join(project_root_path, "ReproduceDPR/Reader/data",
                                          args.dataset, args.prefix_retriever_name, "test.pkl")
            test_dataset = ReaderDataset(
                dataset_file_path=test_file_path, is_eval=True, max_seq_len=args.max_seq_len, max_title_len=args.max_title_len,
                max_answer_len=args.max_answer_len, max_question_len=args.max_question_len, tokenizer=tokenizer,
                num_train_negatives=args.num_train_negatives
            )
            self.test_loader = DataLoader(
                dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
            )

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

        # 模型保存 ==> 完整的reader
        if not os.path.exists(os.path.join(project_root_path, args.saved_model_dir, args.prefix_retriever_name, args.dataset)):
            os.system("mkdir -p %s" % os.path.join(project_root_path, args.saved_model_dir, args.prefix_retriever_name, args.dataset))
            print("阅读器模型保存的目录不存在, 已经自动创建")
        self.writer = SummaryWriter(
            logdir=f"tensorboard/{args.prefix_retriever_name}/{args.dataset}/{prefix}_reader"
        )
        self.model_saved_name = os.path.join(project_root_path, args.saved_model_dir, args.prefix_retriever_name, args.dataset,
                                             f"{prefix}_reader.pth")

    def train(self):
        # 验证集达到最好效果时, 直接验证一波测试集 ==> reader的tensorboard都加上reader前缀 ==> 与其他的区分开
        best_metric = 0.0
        steps = 0
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            for item in self.train_dataloader:
                # batch_chunk
                self.optimizer.zero_grad()
                batch_loss = 0.0
                input_ids, input_mask, input_seg, context_mask, start_label, end_label = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["context_mask"], item["start_label"], item["end_label"]
                input_ids_chunks = torch.chunk(input_ids, chunks=self.args.batch_chunk, dim=0)
                input_mask_chunks = torch.chunk(input_mask, chunks=self.args.batch_chunk, dim=0)
                input_seg_chunks = torch.chunk(input_seg, chunks=self.args.batch_chunk, dim=0)
                context_mask_chunks = torch.chunk(context_mask, chunks=self.args.batch_chunk, dim=0)
                start_label_chunks = torch.chunk(start_label, chunks=self.args.batch_chunk, dim=0)
                end_label_chunks = torch.chunk(end_label, chunks=self.args.batch_chunk, dim=0)
                # 梯度累积步骤
                for chunk in range(len(input_ids_chunks)):
                    single_chunk_loss = self.model(
                        input_ids=input_ids_chunks[chunk].to(self.device), input_mask=input_mask_chunks[chunk].to(self.device),
                        input_seg=input_seg_chunks[chunk].to(self.device), context_mask=context_mask_chunks[chunk].to(self.device),
                        start_label=start_label_chunks[chunk].to(self.device), end_label=end_label_chunks[chunk].to(self.device)
                    ) / len(input_ids_chunks)
                    single_chunk_loss.backward()
                    batch_loss += single_chunk_loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                self.writer.add_scalar("reader/train/loss", batch_loss, global_step=steps)
            # 一个epoch结束, 执行验证流程
            if self.args.eval_start_epoch <= epoch:
                print("%s 当前epoch已经达到验证的时间点, 执行验证集预测" % str(datetime.datetime.now()))
                eval_out = self.temp_eval(is_test=False)
                for p_key in eval_out.keys():
                    for q_key in eval_out[p_key].keys():
                        self.writer.add_scalar(f"reader/valid/{p_key}/{q_key}_em", eval_out[p_key][q_key], global_step=steps)
                if eval_out["under_top50_passages"]["top1"] > best_metric:
                    best_metric = eval_out["under_top50_passages"]["top1"]
                    print("%s || 验证指标Top-50文章下最佳单文本span聚合Top-1的EM值提升至%.3f, 阅读器更新checkpoint点" %
                          (str(datetime.datetime.now()), best_metric))
                    torch.save(self.model.state_dict(), f=self.model_saved_name)
                    if not self.args.not_do_test:
                        print("验证指标提升, 执行测试集预测")
                        test_out = self.temp_eval(is_test=True)
                        for p_key in test_out.keys():
                            for q_key in test_out[p_key].keys():
                                self.writer.add_scalar(f"reader/test/{p_key}/{q_key}_em", test_out[p_key][q_key], global_step=steps)
            else:
                print("%s 当前epoch未达到验证的时间点, 直接保存最新的模型" % str(datetime.datetime.now()))
                torch.save(self.model.state_dict(), f=self.model_saved_name)
        # 使用最优的保存模型重置model, 进行聚合性预测 => 完整性预测相对比较耗时, 时间在6小时~15小时左右, 如果效果不好则不跑这段
        print("模型训练结束, 下面选择最佳参数点进行完整形式的验证以及测试")
        self.model.load_state_dict(torch.load(self.model_saved_name, map_location=self.device), strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("执行验证集的完整性聚合性能测试")
        eval_out = self.full_eval(is_test=False)
        print(eval_out)
        if not self.args.not_do_test:
            print("执行测试集的完整性聚合性能测试")
            test_out = self.full_eval(is_test=True)
            print(test_out)

    def temp_eval(self, is_test: bool):
        n_samples = 0
        match_cnt = {
            "under_top1_passages": {"top1": 0, "top5": 0, "top10": 0},
            "under_top5_passages": {"top1": 0, "top5": 0, "top10": 0},
            "under_top10_passages": {"top1": 0, "top5": 0, "top10": 0},
            "under_top20_passages": {"top1": 0, "top5": 0, "top10": 0},
            "under_top50_passages": {"top1": 0, "top5": 0, "top10": 0},
            "under_top100_passages": {"top1": 0, "top5": 0, "top10": 0}
        }
        dataloader = self.test_loader if is_test else self.dev_loader
        for item in dataloader:
            input_ids, input_mask, input_seg, context_mask, context_range, answers = \
                item["input_ids"], item["input_mask"], item["input_seg"], item["context_mask"], item["context_range"], item["answers"]
            with torch.no_grad():
                match_prob, start_prob, end_prob = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device), context_mask=context_mask.to(self.device)
                )
                _match_prob = match_prob.cpu().numpy()
                _start_prob = start_prob.cpu().numpy()
                _end_prob = end_prob.cpu().numpy()
            _input_ids = input_ids.numpy()
            _answers = answers
            _context_range = context_range
            for k in range(len(_answers)):
                n_samples += 1
                if n_samples % 1000 == 0:
                    print("临时eval已执行%d条" % n_samples)
                context_range = _context_range[k]
                answers = _answers[k]
                match_prob = _match_prob[k]
                start_prob = _start_prob[k]
                end_prob = _end_prob[k]
                input_ids = _input_ids[k]
                # 如果某个index已经处理过了, 直接取其结果
                t_dict = dict()
                for i in [1, 5, 10, 20, 50, 100]:
                    choose_index = np.argmax(match_prob[:i])
                    if choose_index in t_dict.keys():
                        y = t_dict[choose_index]
                    else:
                        x = self.single_passage_decode(
                            match_prob=match_prob[choose_index], start_prob_list=list(start_prob[choose_index]),
                            end_prob_list=list(end_prob[choose_index]), input_ids=list(input_ids[choose_index]),
                            context_start=context_range[choose_index]["start"], context_end=context_range[choose_index]["end"]
                        )
                        r = sorted(x.items(), key=lambda x: x[1], reverse=True)
                        y = self.evaluate_top_n_answers_match(gold_answers=answers, predict_answers=[t[0] for t in r[:10]])
                        t_dict[choose_index] = y
                    if y["top1"]:
                        match_cnt[f"under_top{i}_passages"]["top1"] += 1
                    if y["top5"]:
                        match_cnt[f"under_top{i}_passages"]["top5"] += 1
                    if y["top10"]:
                        match_cnt[f"under_top{i}_passages"]["top10"] += 1
        out = dict()  # 将计数转化为准确率 ==> em
        for p_key in match_cnt.keys():
            out[p_key] = dict()
            for a_key in match_cnt[p_key].keys():
                out[p_key][a_key] = match_cnt[p_key][a_key] / n_samples
        return out

    def full_eval(self, is_test: bool):
        n_samples = 0
        match_cnt = {
            "under_top1_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }, "under_top5_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }, "under_top10_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }, "under_top20_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }, "under_top50_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }, "under_top100_passages": {
                "single_agg_top1": 0, "single_agg_top5": 0, "single_agg_top10": 0, "multi_agg_top1": 0,
                "multi_agg_top5": 0, "multi_agg_top10": 0
            }
        }
        dataloader = self.test_loader if is_test else self.dev_loader
        for item in dataloader:
            input_ids, input_mask, input_seg, context_mask, context_range, answers = \
                item["input_ids"], item["input_mask"], item["input_seg"], item["context_mask"], item["context_range"], item["answers"]
            with torch.no_grad():
                match_prob, start_prob, end_prob = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device), context_mask=context_mask.to(self.device)
                )
                _match_prob = match_prob.cpu().numpy()
                _start_prob = start_prob.cpu().numpy()
                _end_prob = end_prob.cpu().numpy()
            _input_ids = input_ids.numpy()
            _answers = answers
            _context_range = context_range
            for k in range(len(_answers)):
                n_samples += 1
                if n_samples % 100 == 0:
                    print("完整性聚合eval已执行%d条" % n_samples)
                context_range = _context_range[k]
                answers = _answers[k]
                match_prob = _match_prob[k]
                start_prob = _start_prob[k]
                end_prob = _end_prob[k]
                input_ids = _input_ids[k]
                # 为每一个doc都进行解码
                single_agg_tuple_list = []
                multi_agg_map = collections.defaultdict(float)
                for i in range(len(context_range)):
                    x = self.single_passage_decode(
                        match_prob=match_prob[i], start_prob_list=list(start_prob[i]), end_prob_list=list(end_prob[i]),
                        input_ids=list(input_ids[i]), context_start=context_range[i]["start"], context_end=context_range[i]["end"]
                    )
                    single_agg_tuple_list.extend(list(x.items()))
                    for key in x.keys():
                        multi_agg_map[key] += x[key]
                    # 考虑在不同检索召回设定下的em指标
                    if i + 1 in [1, 5, 10, 20, 50, 100]:  # TOP-1, TOP-5, TOP-10, TOP-20, TOP-50, TOP-100
                        for_single = sorted(single_agg_tuple_list, key=lambda x: x[1], reverse=True)
                        for_multi = sorted(multi_agg_map.items(), key=lambda x: x[1], reverse=True)
                        y = self.evaluate_top_n_answers_match(gold_answers=answers, predict_answers=[t[0] for t in for_single[:10]])
                        if y["top1"]:
                            match_cnt[f"under_top{i + 1}_passages"]["single_agg_top1"] += 1
                        if y["top5"]:
                            match_cnt[f"under_top{i + 1}_passages"]["single_agg_top5"] += 1
                        if y["top10"]:
                            match_cnt[f"under_top{i + 1}_passages"]["single_agg_top10"] += 1
                        y = self.evaluate_top_n_answers_match(gold_answers=answers, predict_answers=[t[0] for t in for_multi[:10]])
                        if y["top1"]:
                            match_cnt[f"under_top{i + 1}_passages"]["multi_agg_top1"] += 1
                        if y["top5"]:
                            match_cnt[f"under_top{i + 1}_passages"]["multi_agg_top5"] += 1
                        if y["top10"]:
                            match_cnt[f"under_top{i + 1}_passages"]["multi_agg_top10"] += 1
        out = dict()  # 将计数转化为准确率 ==> em
        for p_key in match_cnt.keys():
            out[p_key] = dict()
            for a_key in match_cnt[p_key].keys():
                out[p_key][a_key] = match_cnt[p_key][a_key] / n_samples
        return out

    def evaluate_top_n_answers_match(self, gold_answers: list, predict_answers: list):
        """
        :param gold_answers: 标注的答案
        :param predict_answers: 预测的答案 => 最多十个
        :return:
        """
        for i, predict_answer in enumerate(predict_answers):
            if self.args.language == "en":
                predict_answer = qa_en_normalize_answer(predict_answer)
            else:
                predict_answer = qa_zn_normalize_answer(predict_answer)
            for gold_answer in gold_answers:
                if self.args.language == "en":
                    if qa_en_normalize_answer(gold_answer) == predict_answer:
                        if i < 1:
                            return {"top1": True, "top5": True, "top10": True}
                        elif i < 5:
                            return {"top1": False, "top5": True, "top10": True}
                        else:
                            return {"top1": False, "top5": False, "top10": True}
                else:
                    if qa_zn_normalize_answer(gold_answer) == predict_answer:
                        if i < 1:
                            return {"top1": True, "top5": True, "top10": True}
                        elif i < 5:
                            return {"top1": False, "top5": True, "top10": True}
                        else:
                            return {"top1": False, "top5": False, "top10": True}
        return {"top1": False, "top5": False, "top10": False}

    def single_passage_decode(self, match_prob: float, start_prob_list: list, end_prob_list: list,
                              context_start: int, context_end: int, input_ids: list):
        """
        :param match_prob: 当前文章的精细化匹配得分
        :param start_prob_list: 作为答案开始的得分
        :param end_prob_list: 作为答案结束的得分
        :param context_start: 解码参考的起始点
        :param context_end: 解码参考的结束点
        :param input_ids: 输入token的ids序列
        :return: 字典 ==> answer: score
        """
        max_span_len = max_span_len_map[self.args.dataset]
        answer_score_map = collections.defaultdict(float)
        for i in range(context_start, context_end + 1):
            for j in range(i, context_end + 1):
                if j - i + 1 > max_span_len:
                    break
                answer = self.span_ids_to_answer_string(input_ids[i:j+1])
                answer_score_map[answer] += 100000 * match_prob * start_prob_list[i] * end_prob_list[j]  # 分数放大
        return answer_score_map

    def span_ids_to_answer_string(self, span_ids: list):
        tokens = self.tokenizer.convert_ids_to_tokens(span_ids)
        out = []
        for token in tokens:
            if token not in ["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]"]:
                if token.startswith("##"):
                    out.append(token[2:])
                else:
                    if self.args.language == "en":
                        out.append(" ")
                    out.append(token)
        return "".join(out).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阅读器")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zn"], help="训练语言")
    parser.add_argument("--prefix_retriever_name", type=str, help="强制参数, 训练这个阅读器所对应的原始检索器是谁")
    parser.add_argument("--dataset", type=str, default="natural_questions", help="数据集名称",
                        choices=["natural_questions", "squad", "cmrc_drcd", "trivia_qa"])
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader的异步加载进程数")
    parser.add_argument("--batch_size", type=int, default=16, help="训练的批大小")
    parser.add_argument("--batch_chunk", type=int, default=8, help="单批次切分成多少个块")
    parser.add_argument("--max_seq_len", type=int, default=384, help="输入文本最大长度")
    parser.add_argument("--max_question_len", type=int, default=30, help="问题最大长度")
    parser.add_argument("--max_title_len", type=int, default=20, help="文章标题最大长度")
    parser.add_argument("--max_answer_len", type=int, default=50, help="训练时最大答案长度")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="进行验证时的batch_size, 照顾RPE")
    parser.add_argument("--eval_start_epoch", type=int, default=15, help="从哪一个epoch开始进行模型的验证")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率大小")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW使用的权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-9, help="最小的学习率")
    parser.add_argument("--warm_up_rate", type=float, default=0.1, help="热启动过程占整个训练过程的比例")
    parser.add_argument("--clip_norm", type=float, default=0.25, help="梯度裁剪最大范数")
    parser.add_argument("--gpu_id", type=int, default=0, help="占用那个GPU显卡")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="最大训练的epoch数")
    parser.add_argument("--num_train_negatives", type=int, default=23, help="训练时的负样本数")
    parser.add_argument("--not_do_test", default=False, action="store_true", help="是否对测试集表现进行验证, 只有SQuAD不需要")

    # 使用的预训练语言模型
    parser.add_argument("--pretrain_model_path", type=str, help="使用的预训练语言模型文件夹",
                        default="ModelStorage/ape_pretrain_models/BERT_Base_EN/")
    # 模型保存路径
    parser.add_argument("--saved_model_dir", type=str, help="retriever检索器模型保存路径",
                        default="ModelStorage/dpr_baseline/reader/")
    args = parser.parse_args()
    print("创建训练对象")
    obj = TrainingFramework(args)
    print("训练对象创建完成, 执行训练过程")
    obj.train()
    obj.writer.close()  # 关闭tensorboard日志写对象
    print("训练任务完成")


