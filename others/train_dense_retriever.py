e"""
@Author: quanlin
@CreateTime: 2021/7/16
@Usage: 训练稠密向量检索模型 Dense Passage Retriever
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, logging
from RPEBERT import RPEBertModel
logging.set_verbosity_error()
from tensorboardX import SummaryWriter
from LRSchedule import WarmUpLinearDecay
from torch import optim
import pickle
import json
import datetime
import numpy as np
from random import shuffle, randint
import argparse

project_root_path = "/home/ldmc/nfs/quanlin/GraduateDesign"  # 整个项目的目录入口
# project_root_path = "/mnt/nfs-storage/quanlin/GraduateDesign"
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


class RetrieverDataset(Dataset):
    """
    检索器训练数据集, 切记, 对于每一个question, 不管取多少负样本, 返回的doc集第一个必须是正样本
    """
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int, dataset_file_path: str,
                 num_hard_negatives: int, num_random_negatives: int, is_random_pick: bool):
        """
        :param tokenizer: 分词器
        :param max_seq_len: 最大长度
        :param dataset_file_path: pkl文件地址
        :param num_random_negatives: 取多少随机负样本
        :param num_hard_negatives: 取多少难负样本
        :param is_random_pick: 是否随机取正样本 and 负样本, 训练随机, 测试不随机
        """
        super(RetrieverDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.is_random_pick = is_random_pick
        with open(dataset_file_path, "rb") as f:
            self.data = pickle.load(f)
            shuffle(self.data)  # load之后随机打乱一次
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index: 索引项
        :return: ids, 不返回torch.tensor形式, 由collate_fn函数实现tensor化
        """
        item = self.data[index]
        # 编码query, doc
        query, pos_docs, random_neg_docs, hard_neg_docs = item["question"], item["positive_ctxs"], \
            item["negative_ctxs"], item["hard_negative_ctxs"]
        # docs的每一项均为列表, 只需要关注"title"项和"text"项
        query_tokens = ["[CLS]"] + self.tokenizer.tokenize(query)[:self.max_seq_len - 2] + ["[SEP]"]
        query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        query_mask = [1.0] * len(query_ids)

        docs_ids, docs_mask = [], []
        if self.is_random_pick:  # 对于训练集对象来说, 随便shuffle无所谓
            shuffle(pos_docs)
            shuffle(random_neg_docs)
            shuffle(hard_neg_docs)
        # 保证第一个样本必定为正样本
        """
        !!!!!!!注意: 正样本必有, 所以不用担心, 随机负样本也必有, 但是难负样本不一定会有, 或者数量满足不了
        -> 解决办法: 不足的从随机负样本里补 
        """
        if len(hard_neg_docs) < self.num_hard_negatives:
            pick_docs = pos_docs[:1] + hard_neg_docs[:self.num_hard_negatives]
            extra = self.num_hard_negatives - len(hard_neg_docs) + self.num_random_negatives
            pick_docs += random_neg_docs[:extra]
            extra = self.num_random_negatives + self.num_hard_negatives + 1 - len(pick_docs)
            for i in range(extra):
                pick_docs.append(random_neg_docs[randint(0, len(random_neg_docs) - 1)])
        else:
            pick_docs = pos_docs[:1] + random_neg_docs[:self.num_random_negatives] + hard_neg_docs[:self.num_hard_negatives]
        for doc_item in pick_docs:
            title, text = doc_item["title"], doc_item["text"]
            title_tokens = ["[CLS]"] + self.tokenizer.tokenize(title) + ["[SEP]"]
            text_tokens = self.tokenizer.tokenize(text)[:self.max_seq_len - len(title_tokens) - 1] + ["[SEP]"]
            doc_tokens = title_tokens + text_tokens
            docs_ids.append(self.tokenizer.convert_tokens_to_ids(doc_tokens))
            docs_mask.append([1.0] * len(doc_tokens))
        return {
            "query_ids": query_ids, "query_mask": query_mask,
            "docs_ids": docs_ids, "docs_mask": docs_mask
        }


def collate_fn(batch):
    """
    :param batch: 一次批次的数据, 需要处理query和doc
    :return: 统一query和doc序列长度的输入
    """
    max_query_len, max_doc_len = 0, 0
    for item in batch:
        # 先检查query
        if len(item["query_ids"]) > max_query_len:
            max_query_len = len(item["query_ids"])
        for doc_ids in item["docs_ids"]:
            if len(doc_ids) > max_doc_len:
                max_doc_len = len(doc_ids)
    # 根据最大长度进行padding
    query_ids, query_mask, docs_ids, docs_mask = [], [], [], []
    for item in batch:
        extra = max_query_len - len(item["query_ids"])
        if extra > 0:
            query_ids.append(item["query_ids"] + [0] * extra)
            query_mask.append(item["query_mask"] + [0.0] * extra)
        else:
            query_ids.append(item["query_ids"])
            query_mask.append(item["query_mask"])
        # doc比较特殊
        for i in range(len(item["docs_ids"])):
            extra = max_doc_len - len(item["docs_ids"][i])
            if extra > 0:
                docs_ids.append(item["docs_ids"][i] + [0] * extra)
                docs_mask.append(item["docs_mask"][i] + [0.0] * extra)
            else:
                docs_ids.append(item["docs_ids"][i])
                docs_mask.append(item["docs_mask"][i])
    return {
        "query_ids": torch.tensor(query_ids).long(), "query_mask": torch.tensor(query_mask).float(),
        "docs_ids": torch.tensor(docs_ids).long(), "docs_mask": torch.tensor(docs_mask).float()
    }


class RPEModelWithPooler(torch.nn.Module):
    def __init__(self, pretrain_model_path: str, rpe_method: str, max_seq_len: int):
        super(RPEModelWithPooler, self).__init__()
        with open(os.path.join(pretrain_model_path, "config.json"), "r", encoding="UTF-8") as f:
            config = json.load(f)
        self.rpe_model = RPEBertModel(config=config, rel_method=rpe_method, max_seq_len=max_seq_len)
        # 加载预训练参数
        self.rpe_model.load_state_dict(torch.load(os.path.join(pretrain_model_path, "pytorch_model.bin"), map_location="cpu"))
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(in_features=config["hidden_dim"], out_features=config["hidden_dim"]),
            torch.nn.Tanh()
        )

    def forward(self, input_ids, input_mask):
        rep = self.rpe_model(input_ids, input_mask)[:, 0, :]  # (bsz, dim)
        return self.pooler(rep)


class BiEncoderRetriever(torch.nn.Module):
    def __init__(self, pretrain_model_path: str, share_parameters=False, is_rpe=False,
                 rpe_method="", max_seq_len=256):
        """
        :param pretrain_model_path: 预训练语言模型文件夹
        :param share_parameters: query侧和doc侧的文本encoder是否共享参数, 默认不共享
        :param is_rpe: 是否是相对位置编码
        :param max_seq_len: 最大文本长度 ==> 仅仅RPE模型会使用
        :param rpe_method: 相对位置编码方法
        """
        super(BiEncoderRetriever, self).__init__()
        self.is_rpe = is_rpe
        if share_parameters:
            if is_rpe:
                bert_encoder = RPEModelWithPooler(pretrain_model_path, rpe_method, max_seq_len)
            else:
                bert_encoder = BertModel.from_pretrained(pretrain_model_path)
            self.query_encoder = bert_encoder
            self.doc_encoder = bert_encoder
        else:
            if is_rpe:
                self.query_encoder = RPEModelWithPooler(pretrain_model_path, rpe_method, max_seq_len)
                self.doc_encoder = RPEModelWithPooler(pretrain_model_path, rpe_method, max_seq_len)
            else:
                self.query_encoder = BertModel.from_pretrained(pretrain_model_path)
                self.doc_encoder = BertModel.from_pretrained(pretrain_model_path)

    def forward(self, input_query_ids, input_query_mask, input_docs_ids, input_docs_mask, return_nll_loss=True):
        """
        :param input_query_ids: 问题的ids, shape: (q_bsz, q_seq_len)
        :param input_query_mask: 问题的mask, shape: (q_bsz, q_seq_len)
        :param input_docs_ids: 文章的ids, shape: (d_bsz, d_seq_len)
        :param input_docs_mask: 文章的ids, shape: (d_bsz, d_seq_len)
        :param return_nll_loss: 为True时, 计算nll_loss; 为False时, 返回query和doc的representations
        :return: 当return_nll_loss为True时, 返回nll_loss; 当return_nll_loss为False时, 返回query和doc的表征
        """
        # 检查doc的bsz是否与预计一致
        q_bsz, d_bsz = input_query_ids.size(0), input_docs_ids.size(0)
        assert d_bsz % q_bsz == 0  # 必须可以整除
        num_docs_per_query = d_bsz // q_bsz
        if self.is_rpe:
            query_rep = self.query_encoder(input_query_ids, input_query_mask)  # (q_bsz, dim)
            doc_rep = self.doc_encoder(input_docs_ids, input_docs_mask)  # (d_bsz, dim)
        else:
            query_rep = self.query_encoder(input_query_ids, input_query_mask).pooler_output  # (q_bsz, dim)
            doc_rep = self.doc_encoder(input_docs_ids, input_docs_mask).pooler_output  # (d_bsz, dim)
        if return_nll_loss:
            inner_product = torch.einsum("qd,pd->qp", query_rep, doc_rep)  # (q_bsz, d_bsz)
            log_probability = torch.log_softmax(inner_product, dim=-1)
            pos_index = torch.arange(start=0, end=q_bsz, device=input_query_ids.device).long()[:, None] * num_docs_per_query
            pos_log_probability = torch.gather(log_probability, dim=1, index=pos_index)  # (q_bsz, 1)
            nll_loss = -torch.mean(pos_log_probability)
            return nll_loss
        else:
            return query_rep, doc_rep


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
        # 模型初始化
        self.model = BiEncoderRetriever(
            pretrain_model_path=os.path.join(project_root_path, args.pretrain_model_path),
            share_parameters=args.share_parameters, is_rpe=is_rpe, rpe_method=rpe_method,
            max_seq_len=args.max_seq_len
        )

        # 训练/验证集, 验证集分为NLL模式的验证集和AVG模式的验证集
        tokenizer = BertTokenizer(vocab_file=os.path.join(project_root_path, args.pretrain_model_path, "vocab.txt"))
        if args.dataset == "natural_questions":
            train_file_path = os.path.join(project_root_path, "data/qa_data/NaturalQuestions/nq-retriever-train.pkl")
            dev_file_path = os.path.join(project_root_path, "data/qa_data/NaturalQuestions/nq-retriever-dev.pkl")
        elif args.dataset == "squad":
            train_file_path = os.path.join(project_root_path, "data/qa_data/SQuAD/squad-retriever-train.pkl")
            dev_file_path = os.path.join(project_root_path, "data/qa_data/SQuAD/squad-retriever-dev.pkl")
        elif args.dataset == "trivia_qa":
            train_file_path = os.path.join(project_root_path, "data/qa_data/TriviaQA/trivia-retriever-train.pkl")
            dev_file_path = os.path.join(project_root_path, "data/qa_data/TriviaQA/trivia-retriever-dev.pkl")
        else:
            train_file_path = os.path.join(project_root_path, "data/qa_data/CMRC_DRCD/cmrc_drcd-retriever-train.pkl")
            dev_file_path = os.path.join(project_root_path, "data/qa_data/CMRC_DRCD/cmrc_drcd-retriever-dev.pkl")
        train_dataset = RetrieverDataset(
            max_seq_len=args.max_seq_len, tokenizer=tokenizer, dataset_file_path=train_file_path, is_random_pick=True,
            num_hard_negatives=args.train_hard_negatives, num_random_negatives=args.train_random_negatives
        )
        nll_dev_dataset = RetrieverDataset(
            max_seq_len=args.max_seq_len, tokenizer=tokenizer, dataset_file_path=dev_file_path, is_random_pick=False,
            num_hard_negatives=args.train_hard_negatives, num_random_negatives=args.train_random_negatives
        )
        avg_rank_dev_dataset = RetrieverDataset(
            max_seq_len=args.max_seq_len, tokenizer=tokenizer, dataset_file_path=dev_file_path, is_random_pick=False,
            num_hard_negatives=args.val_avg_rank_hard_neg, num_random_negatives=args.val_avg_rank_random_neg
        )
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn
        )
        self.nll_dev_loader = DataLoader(
            dataset=nll_dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn
        )
        self.avg_rank_dev_loader = DataLoader(
            dataset=avg_rank_dev_dataset, batch_size=args.dev_avg_rank_batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn
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

        # 模型保存, 日志记录文件等琐碎项配置
        # 模型保存目录不存在, 自动创建
        if not os.path.exists(os.path.join(project_root_path, args.saved_model_dir, args.dataset)):
            os.system("mkdir -p %s" % os.path.join(project_root_path, args.saved_model_dir, args.dataset))
            print("模型保存的目录不存在, 已经自动创建")
        if args.share_parameters:
            self.writer = SummaryWriter(
                logdir="tensorboard/{dataset}/{prefix}_bi-encoder-share".format(prefix=prefix, dataset=args.dataset)
            )
            self.query_model_saved_name = os.path.join(
                project_root_path, args.saved_model_dir, args.dataset, "%s_bi-encoder-share.pth" % prefix)
            self.doc_model_saved_name = self.query_model_saved_name
        else:
            self.writer = SummaryWriter(
                logdir="tensorboard/{dataset}/{prefix}_bi-encoder-not-share".format(prefix=prefix, dataset=args.dataset)
            )
            self.query_model_saved_name = os.path.join(project_root_path, args.saved_model_dir, args.dataset,
                                                       "%s_bi-encoder-query.pth" % prefix)
            self.doc_model_saved_name = os.path.join(project_root_path, args.saved_model_dir, args.dataset,
                                                     "%s_bi-encoder-doc.pth" % prefix)

    def train(self):
        steps = 0
        best_metric = 65535  # 越小越好, 注意在avg_rank评价奏效的时候, 直接执行硬性切换
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            for item in self.train_loader:
                self.optimizer.zero_grad()
                query_ids, query_mask, docs_ids, docs_mask = item["query_ids"], item["query_mask"], item["docs_ids"], item["docs_mask"]
                nll_loss = self.model.forward(
                    input_query_ids=query_ids.to(self.device), input_query_mask=query_mask.to(self.device),
                    input_docs_ids=docs_ids.to(self.device), input_docs_mask=docs_mask.to(self.device)
                )
                nll_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                self.writer.add_scalar("train/nll_loss", nll_loss.item(), global_step=steps)
            # 进行验证
            dev_nll_loss = self.valid_nll_loss()
            if epoch < self.args.val_avg_rank_start_epoch:
                dev_avg_rank, recall_dic = 0, {"r1": 0, "r5": 0, "r10": 0, "r20": 0, "r50": 0, "r100": 0}
            else:
                dev_avg_rank, recall_dic = self.valid_avg_rank()
            self.writer.add_scalar("valid/nll_loss", dev_nll_loss, global_step=steps)
            self.writer.add_scalar("valid/avg_rank", dev_avg_rank, global_step=steps)
            self.writer.add_scalar("valid/R@1", recall_dic["r1"], global_step=steps)
            self.writer.add_scalar("valid/R@5", recall_dic["r5"], global_step=steps)
            self.writer.add_scalar("valid/R@10", recall_dic["r10"], global_step=steps)
            self.writer.add_scalar("valid/R@20", recall_dic["r20"], global_step=steps)
            self.writer.add_scalar("valid/R@50", recall_dic["r50"], global_step=steps)
            self.writer.add_scalar("valid/R@100", recall_dic["r100"], global_step=steps)
            print("[Time: {}] || [Epoch: {}] || [Valid NLL Loss: {:.5f}] || [Valid Avg Rank: {:.5f}]".format(
                datetime.datetime.now(), epoch + 1, dev_nll_loss, dev_avg_rank
            ))
            print("||R@1: %.3f ||R@5: %.3f ||R@10: %.3f ||R@20: %.3f ||R@50: %.3f ||R@100: %.3f" % (
                recall_dic["r1"], recall_dic["r5"], recall_dic["r10"], recall_dic["r20"], recall_dic["r50"],
                recall_dic["r100"]
            ))
            if epoch < self.args.val_avg_rank_start_epoch:
                if dev_nll_loss < best_metric:
                    save_flag = True
                    best_metric = dev_nll_loss
                else:
                    save_flag = False
                print("当前选择的验证metric为NLL-Loss, 最佳值为: %.5f" % best_metric)
            elif epoch == self.args.val_avg_rank_start_epoch:
                best_metric = dev_avg_rank
                # best_metric = recall_dic["r100"]
                save_flag = True
                print("验证metric开始切换为Avg-Rank, 最佳值为: %.5f" % best_metric)
                # print("验证metric开始切换为R@100, 最佳值为: %.5f" % best_metric)
            else:
                # if recall_dic["r100"] > best_metric:
                #     save_flag = True
                #     best_metric = recall_dic["r100"]
                if dev_avg_rank > best_metric:
                    save_flag = True
                    best_metric = dev_avg_rank
                else:
                    save_flag = False
                print("当前选择的验证metric为Avg-Rank, 最佳值为: %.5f" % best_metric)
                # print("验证metric开始切换为R@100, 最佳值为: %.5f" % best_metric)
            print("================================================")
            if save_flag:
                if self.args.share_parameters:
                    torch.save(self.model.query_encoder.state_dict(), f=self.query_model_saved_name)
                else:
                    torch.save(self.model.query_encoder.state_dict(), f=self.query_model_saved_name)
                    torch.save(self.model.doc_encoder.state_dict(), f=self.doc_model_saved_name)

    def valid_nll_loss(self):
        self.model.eval()
        sum_loss, count = 0.0, 0
        for item in self.nll_dev_loader:
            query_ids, query_mask, docs_ids, docs_mask = item["query_ids"], item["query_mask"], item["docs_ids"], item["docs_mask"]
            with torch.no_grad():
                nll_loss = self.model.forward(
                    input_query_ids=query_ids.to(self.device), input_query_mask=query_mask.to(self.device),
                    input_docs_ids=docs_ids.to(self.device), input_docs_mask=docs_mask.to(self.device)
                )
                count += query_ids.size(0)
                sum_loss += nll_loss.item() * query_ids.size(0)
        self.model.train()
        return sum_loss / count

    def valid_avg_rank(self):
        """
        与原作者的代码实现有区别, 我们这里算的是倒数, 其实可以看成IR中常用的MRR指标
        """
        self.model.eval()
        all_query_vectors, all_doc_vectors = [], []
        for item in self.avg_rank_dev_loader:
            query_ids, query_mask, docs_ids, docs_mask = item["query_ids"], item["query_mask"], item["docs_ids"], item["docs_mask"]
            with torch.no_grad():
                query_rep, doc_rep = self.model.forward(
                    input_query_ids=query_ids.to(self.device), input_query_mask=query_mask.to(self.device),
                    input_docs_ids=docs_ids.to(self.device), input_docs_mask=docs_mask.to(self.device),
                    return_nll_loss=False
                )
                query_rep, doc_rep = query_rep.cpu().numpy(), doc_rep.cpu().numpy()
                all_query_vectors.extend(list(query_rep))
                all_doc_vectors.extend(list(doc_rep))
        inner_product_similarity = np.dot(np.array(all_query_vectors), np.array(all_doc_vectors).T)  # (q, p)
        num_questions = len(all_query_vectors)
        sum_rank = 0
        # 额外计算recall@K
        recall_1, recall_5, recall_10, recall_20, recall_50, recall_100 = 0, 0, 0, 0, 0, 0
        for i in range(num_questions):
            ips = inner_product_similarity[i]
            pos_ip = ips[i * (1 + self.args.val_avg_rank_hard_neg + self.args.val_avg_rank_random_neg)]
            rank = np.sum(ips > pos_ip) + 1
            if rank <= 1:
                recall_1 += 1
            if rank <= 5:
                recall_5 += 1
            if rank <= 10:
                recall_10 += 1
            if rank <= 20:
                recall_20 += 1
            if rank <= 50:
                recall_50 += 1
            if rank <= 100:
                recall_100 += 1
            sum_rank += 1.0 / rank
        self.model.train()
        return sum_rank / num_questions, {
            "r1": recall_1 / num_questions, "r5": recall_5 / num_questions, "r10": recall_10 / num_questions,
            "r20": recall_20 / num_questions, "r50": recall_50 / num_questions, "r100": recall_100 / num_questions
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于Bi-Encoder模式和DPR的训练方式构建稠密向量检索器")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zn"], help="训练语言")
    parser.add_argument("--dataset", type=str, default="natural_questions", help="数据集名称",
                        choices=["natural_questions", "squad", "cmrc_drcd", "trivia_qa"])
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader的异步加载进程数")
    parser.add_argument("--share_parameters", default=False, action="store_true", help="Query侧和Doc侧Encoder是否共享参数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练的批大小")
    parser.add_argument("--dev_avg_rank_batch_size", type=int, default=8, help="进行avg_rank时的batch_size, 相对小一点, 为了照顾RPE")
    parser.add_argument("--max_seq_len", type=int, default=256, help="输入文本最大长度")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率大小")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW使用的权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-9, help="最小的学习率")
    parser.add_argument("--warm_up_rate", type=float, default=0.1, help="热启动过程占整个训练过程的比例")
    parser.add_argument("--clip_norm", type=float, default=0.25, help="梯度裁剪最大范数")
    parser.add_argument("--gpu_id", type=int, default=0, help="占用那个GPU显卡")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="最大训练的epoch数")
    parser.add_argument("--train_hard_negatives", type=int, default=1, help="训练时取多少的难负样本, 也是NLL验证设置的值")
    parser.add_argument("--train_random_negatives", type=int, default=0, help="训练时取多少的随机负样本, 也是NLL验证设置的值")
    parser.add_argument("--val_avg_rank_hard_neg", type=int, default=30, help="使用Avg-Rank验证时每个Q取多少的难负样本, 不shuffle")
    parser.add_argument("--val_avg_rank_random_neg", type=int, default=30, help="使用Avg-Rank验证时每个Q取多少随机负样本, 不shuffle")
    parser.add_argument("--val_avg_rank_start_epoch", type=int, default=10, help="从第几个epoch开始使用Avg-Rank验证")

    # 使用的预训练语言模型
    parser.add_argument("--pretrain_model_path", type=str, help="使用的预训练语言模型文件夹",
                        default="ModelStorage/ape_pretrain_models/BERT_Base_EN/")
    # 模型保存路径
    parser.add_argument("--saved_model_dir", type=str, help="retriever检索器模型保存路径",
                        default="ModelStorage/dpr_baseline/retriever/")
    # 备注: 具体模型名, tensorboard日志名自行根据数据集、参数等设置, 具有动态性

    args = parser.parse_args()
    print("创建训练对象")
    obj = TrainingFramework(args)
    print("训练对象创建完成, 执行训练过程")
    obj.train()
    obj.writer.close()  # 关闭tensorboard日志写对象
    print("训练任务完成")




