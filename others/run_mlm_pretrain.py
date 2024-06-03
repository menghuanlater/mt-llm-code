"""
@Author: quanlin
@CreateTime: 2021/7/27
@Usage: 执行预训练
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import BertTokenizer
from RPEBERT import RPEBertModel
from LRSchedule import WarmUpLinearDecay
from tensorboardX import SummaryWriter
from collections import OrderedDict
import argparse
import numpy as np
import json
import pickle
import os

project_root_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/语言模型预训练实验"
corpus_length_map = {
    "en": {"train": 40501092, "dev": 40000},
    "zn": {"train": 38528338, "dev": 40000}
}


class PretrainDataset(Dataset):
    """
    与正常的Dataset不同, 这里数据集是存储在多个文件中的, 防止数据全部加载入内存导致内存爆炸(尤其在线上docker实例内存有限的情况下)
    shuffle全部取消; tokenizer仅仅负责转ids, 所有token均已提前切分好
    """
    def __init__(self, tokenizer: BertTokenizer, language: str, is_eval: bool, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.language = language
        if language == "en":
            if is_eval:
                self.prefix = os.path.join(project_root_path, "data/en_pretrain_corpus/final_version/valid")
                self.dataset_length = corpus_length_map[language]["dev"]
            else:
                self.prefix = os.path.join(project_root_path, "data/en_pretrain_corpus/final_version/train")
                self.dataset_length = corpus_length_map[language]["train"]
        else:
            if is_eval:
                self.prefix = os.path.join(project_root_path, "data/zn_pretrain_corpus/final_version/valid")
                self.dataset_length = corpus_length_map[language]["dev"]
            else:
                self.prefix = os.path.join(project_root_path, "data/zn_pretrain_corpus/final_version/train")
                self.dataset_length = corpus_length_map[language]["train"]

        # 动态加载数据文件, 两个指示标, 表示每个数据文件所对应的索引的起始和结尾(闭区间)
        self.range_dict = json.load(open(os.path.join(self.prefix, "range_dict.json"), "r", encoding="UTF-8"))
        self.left_edge, self.right_edge = -1, -1
        self.data = None

    def __len__(self):
        return self.dataset_length

    def update(self, index):
        # 更新文件
        if self.data is not None:
            self.data.clear()
        x = None
        for key in self.range_dict.keys():
            if self.range_dict[key]["left_edge"] <= index <= self.range_dict[key]["right_edge"]:
                x = self.range_dict[key]
                break
        self.left_edge, self.right_edge = x["left_edge"], x["right_edge"]
        with open(os.path.join(self.prefix, x["file_name"]), "rb") as f:
            self.data = pickle.load(f)
            assert len(self.data) == self.right_edge - self.left_edge + 1

    def __getitem__(self, index):
        if index > self.right_edge:
            self.update(index)
        item = self.data[index - self.left_edge]
        A_original_tokens, A_mask_tokens, A_label_mask, B_original_tokens, B_mask_tokens, B_label_mask = \
            item["A_original_tokens"], item["A_mask_tokens"], item["A_label_mask"], item["B_original_tokens"], \
            item["B_mask_tokens"], item["B_label_mask"]
        # 进行整理, 都是字符串, 需要转化为列表
        A_original_tokens = A_original_tokens.split("\t")
        A_mask_tokens = A_mask_tokens.split("\t")
        B_original_tokens = B_original_tokens.split("\t")[:126]
        B_mask_tokens = B_mask_tokens.split("\t")[:126]
        B_label_mask = B_label_mask[:126]
        mask_tokens = ["[CLS]"] + A_mask_tokens + ["[SEP]"] + B_mask_tokens + ["[SEP]"]
        origin_tokens = ["[CLS]"] + A_original_tokens + ["[SEP]"] + B_original_tokens + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(mask_tokens)
        mask = [1] * len(ids)
        seg = [0] * (len(A_mask_tokens) + 2) + [1] * (len(B_mask_tokens) + 1)
        label = self.tokenizer.convert_tokens_to_ids(origin_tokens)
        label_mask = [0]
        for x in A_label_mask:
            if x == "1":
                label_mask.append(1)
            else:
                label_mask.append(0)
        label_mask.append(0)
        for x in B_label_mask:
            if x == "1":
                label_mask.append(1)
            else:
                label_mask.append(0)
        label_mask.append(0)
        extra = self.max_seq_len - len(ids)
        if extra > 0:
            ids += [0] * extra
            mask += [0] * extra
            seg += [1] * extra
            label += [0] * extra
            label_mask += [0] * extra
        return {
            "input_ids": torch.tensor(ids).long(), "input_mask": torch.tensor(mask).float(),
            "input_seg": torch.tensor(seg).long(), "label": torch.tensor(label).long(),
            "label_mask": torch.tensor(label_mask).long()
        }


class MLMPretrainModel(torch.nn.Module):
    def __init__(self, config: dict, rpe_method: str, max_seq_len: int):
        super(MLMPretrainModel, self).__init__()
        self.vocab_size = config["vocab_size"]
        self.rpe_bert = RPEBertModel(config=config, rel_method=rpe_method, max_seq_len=max_seq_len)
        self.rep_transform = torch.nn.Linear(in_features=config["hidden_dim"], out_features=config["hidden_dim"])
        self.rep_layer_norm = torch.nn.LayerNorm(config["hidden_dim"], eps=config["layer_norm_eps"])
        self.decoder = torch.nn.Linear(in_features=config["hidden_dim"], out_features=config["vocab_size"])
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids, input_mask, input_seg, label, label_mask):
        """
        :param input_ids: 输入文本序列 (bsz, seq_len)
        :param input_mask: 序列mask (bsz, seq_len)
        :param input_seg: 段id (bsz, seq_len)
        :param label: 标签 (bsz, seq_len)  => 其实就是input_ids未被mask前的序列
        :param label_mask: 标签mask (bsz, seq_len)  => 哪些需要预测, 哪些不需要预测
        :return: MLM平均损失, 当前batch真正需要预测label个数
        """
        enc_rep = self.rpe_bert(input_ids, input_mask, input_seg)
        enc_rep = self.rep_layer_norm(self.rep_transform(enc_rep))
        logits = self.decoder(enc_rep)
        label = label.view(-1)
        whole_seq_loss = self.loss_function(input=logits.view(-1, self.vocab_size), target=label)
        label_mask = label_mask.view(-1)
        num_predict = torch.sum(label_mask)  # 一共有多少是需要预测的
        real_loss = torch.sum(whole_seq_loss * label_mask)  # 总序列真实需要的损失
        predict = torch.argmax(logits, dim=-1).view(-1)  # (bsz * seq_len,)
        num_positive_predict = torch.sum((predict == label) * label_mask)
        return real_loss / num_predict, num_predict, num_positive_predict


class TrainingFramework(object):
    def __init__(self, args):
        self.args = args
        self.device = "cuda:%d" % args.gpu_id

        # 模型
        if self.args.rpe_method == "t5":
            config = json.load(open(os.path.join(
                project_root_path, f"RelPosBERT/configs/t5_{args.language}_config.json"), "r", encoding="UTF-8")
            )
            if self.args.language == "en":
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/T5_Base_EN/pytorch_model.bin"
                )
            else:
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/T5_Base_ZN/pytorch_model.bin"
                )
        elif self.args.rpe_method == "shaw":
            config = json.load(open(os.path.join(
                project_root_path, f"RelPosBERT/configs/shaw_{args.language}_config.json"), "r", encoding="UTF-8")
            )
            if self.args.language == "en":
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/Shaw_Base_EN/pytorch_model.bin"
                )
            else:
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/Shaw_Base_ZN/pytorch_model.bin"
                )
        elif self.args.rpe_method == "lfhc":
            config = json.load(open(os.path.join(
                project_root_path, f"RelPosBERT/configs/lfhc_{args.language}_config.json"), "r", encoding="UTF-8")
            )
            if self.args.language == "en":
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/LFHC_Base_EN/pytorch_model.bin"
                )
            else:
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/LFHC_Base_ZN/pytorch_model.bin"
                )
        elif self.args.rpe_method == "xl":
            config = json.load(open(os.path.join(
                project_root_path, f"RelPosBERT/configs/xl_{args.language}_config.json"), "r", encoding="UTF-8")
            )
            if self.args.language == "en":
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/XL_Base_EN/pytorch_model.bin"
                )
            else:
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/XL_Base_ZN/pytorch_model.bin"
                )
        elif self.args.rpe_method == "gcdf":
            config = json.load(open(os.path.join(
                project_root_path, f"RelPosBERT/configs/gcdf_{args.language}_config.json"), "r", encoding="UTF-8")
            )
            if self.args.language == "en":
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/GCDF_Base_EN/pytorch_model.bin"
                )
            else:
                self.rpe_model_saved_file_name = os.path.join(
                    project_root_path, "ModelStorage/rpe_pretrain_models/GCDF_Base_ZN/pytorch_model.bin"
                )
        else:
            raise Exception("Unknown Relative Position Encoding")
        self.model = MLMPretrainModel(config=config, rpe_method=self.args.rpe_method, max_seq_len=args.max_seq_len)
        self.model.load_state_dict(self.get_ape_pretrain_weights_state_dict(), strict=False)

        # 训练和验证数据集build
        if args.language == "en":
            tokenizer = BertTokenizer(vocab_file=os.path.join(project_root_path, "ModelStorage/ape_pretrain_models/BERT_Base_EN/vocab.txt"))
        else:
            tokenizer = BertTokenizer(vocab_file=os.path.join(project_root_path, "ModelStorage/ape_pretrain_models/RoBERTa_Base_ZN/vocab.txt"))
        train_dataset = PretrainDataset(tokenizer=tokenizer, language=args.language, is_eval=False, max_seq_len=args.max_seq_len)
        dev_dataset = PretrainDataset(tokenizer=tokenizer, language=args.language, is_eval=True, max_seq_len=args.max_seq_len)
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        self.dev_loader = DataLoader(dataset=dev_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)
        all_steps = int(args.num_train_epochs * train_dataset.dataset_length / args.batch_size)
        self.schedule = WarmUpLinearDecay(
            optimizer=self.optimizer, init_rate=args.lr, warm_up_steps=int(args.warm_up_rate * all_steps),
            decay_steps=int((1.0 - args.warm_up_rate) * all_steps), min_lr_rate=args.min_lr
        )
        self.model.to(self.device)
        self.writer = SummaryWriter(logdir=f"tensorboard/{args.language}/{args.rpe_method}")

    def get_ape_pretrain_weights_state_dict(self):
        if self.args.language == "en":
            _ape_state_dict = torch.load(os.path.join(
                project_root_path, "ModelStorage/ape_pretrain_models/BERT_Base_EN/pytorch_model.bin"))
            # 替换所有的gamma和beta
            ape_state_dict = dict()
            for key in _ape_state_dict.keys():
                if "gamma" in key:
                    ape_state_dict[key.replace("gamma", "weight")] = _ape_state_dict[key]
                elif "beta" in key:
                    ape_state_dict[key.replace("beta", "bias")] = _ape_state_dict[key]
                else:
                    ape_state_dict[key] = _ape_state_dict[key]
        else:
            ape_state_dict = torch.load(os.path.join(
                project_root_path, "ModelStorage/ape_pretrain_models/RoBERTa_Base_ZN/pytorch_model.bin"))
        f = open(os.path.join(project_root_path, "RelPosBERT/configs/ape_map_to_rpe.json"), "r", encoding="UTF-8")
        ape_map_to_rpe = json.load(f)
        state_dict = OrderedDict()
        for key in ape_map_to_rpe.keys():
            state_dict[key] = ape_state_dict[ape_map_to_rpe[key]]
        return state_dict

    def train(self):
        steps = 0
        best_ppl = 99999
        for epoch in range(self.args.num_train_epochs):
            for item in self.train_loader:
                self.optimizer.zero_grad()
                input_ids, input_mask, input_seg, label, label_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["label"], item["label_mask"]
                loss, num_predict, num_pos_predict = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device), label=label.to(self.device),
                    label_mask=label_mask.to(self.device)
                )
                if num_predict == 0:
                    num_predict = 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                self.writer.add_scalar("train/ppl", np.exp(loss.item()), global_step=steps)
                self.writer.add_scalar("train/acc", num_pos_predict / num_predict, global_step=steps)
                if steps % self.args.eval_interval == 0:
                    ppl, acc = self.eval()
                    self.writer.add_scalar("valid/ppl", ppl, global_step=steps)
                    self.writer.add_scalar("valid/acc", acc, global_step=steps)
                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save(self.model.state_dict(), f=self.rpe_model_saved_file_name)
                        print("===验证指标提升,预训练模型已经保存在指定目录===")
        self.model.train()

    def eval(self):
        self.model.eval()
        sum_loss, sum_cnt, sum_pos_cnt = 0, 0, 0
        for item in self.dev_loader:
            input_ids, input_mask, input_seg, label, label_mask = \
                item["input_ids"], item["input_mask"], item["input_seg"], item["label"], item["label_mask"]
            with torch.no_grad():
                loss, cnt, pos_cnt = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device), label=label.to(self.device),
                    label_mask=label_mask.to(self.device)
                )
                sum_cnt += cnt.item()
                sum_loss += cnt.item() * loss.item()
                sum_pos_cnt += pos_cnt
        self.model.train()
        return np.exp(sum_loss / sum_cnt), sum_pos_cnt / sum_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行RPEBERT预训练")
    parser.add_argument("--language", type=str, default="zn", choices=["en", "zn"], help="训练语言")
    parser.add_argument("--num_workers", type=int, default=5, help="异步数据加载进程数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")  # 待进行压测
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--rpe_method", type=str, default="gcdf", choices=["t5", "shaw", "lfhc", "xl", "gcdf"], help="相对位置编码方法")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW使用的权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-9, help="最小的学习率")
    parser.add_argument("--warm_up_rate", type=float, default=0.1, help="热启动过程占整个训练过程的比例")
    parser.add_argument("--clip_norm", type=float, default=0.25, help="梯度裁剪最大范数")
    parser.add_argument("--gpu_id", type=int, default=0, help="占用那个GPU显卡")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练epoch数")
    parser.add_argument("--eval_interval", type=int, default=10000, help="模型验证一次间隔")

    args = parser.parse_args()
    print("创建预训练对象")
    obj = TrainingFramework(args)
    print("预训练对象创建完成, 执行预训练过程")
    obj.train()
    obj.writer.close()  # 关闭tensorboard日志写对象
    print("预训练任务完成, 开始新的旅程吧-_-*_*~_~^_^")


