"""
@Author: quanlin03
@CreateTime: 2021/7/27
@Usage: 恢复型预训练, 由于之前训练GCDF_RPE_BERT未保留任务层参数, 导致在下游纠错任务上效果有损失, 所以这次从之前的参数启动,
以之前的预训练数据重新训练, 恢复任务层参数
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import BertTokenizer
from RPEBERT import RPEBertModel
from LRSchedule import WarmUpLinearDecay
import datetime
import argparse
import numpy as np
import json
import joblib
import os

project_root_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/语言模型预训练实验"
work_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03"  # 总工作目录
corpus_length_map = {
    "train": 38528338, "dev": 40000
}


class PretrainDataset(Dataset):
    """
    与正常的Dataset不同, 这里数据集是存储在多个文件中的, 防止数据全部加载入内存导致内存爆炸(尤其在线上docker实例内存有限的情况下)
    shuffle全部取消; tokenizer仅仅负责转ids, 所有token均已提前切分好
    """
    def __init__(self, tokenizer: BertTokenizer, is_eval: bool, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if is_eval:
            self.prefix = os.path.join(project_root_path, "中文mlm预训练数据集/final_version/valid")
            self.dataset_length = corpus_length_map["dev"]
        else:
            self.prefix = os.path.join(project_root_path, "中文mlm预训练数据集/final_version/train")
            self.dataset_length = corpus_length_map["train"]

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
            self.data = joblib.load(f)
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
    def __init__(self, config: dict, max_seq_len: int):
        super(MLMPretrainModel, self).__init__()
        self.vocab_size = config["vocab_size"]
        self.bert_encoder = RPEBertModel(config=config, rel_method="gcdf", max_seq_len=max_seq_len)
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
        enc_rep = self.bert_encoder(input_ids, input_mask, input_seg)
        enc_rep = self.rep_layer_norm(self.rep_transform(enc_rep))
        logits = self.decoder(enc_rep)
        label = label.view(-1)
        whole_seq_loss = self.loss_function(input=logits.view(-1, self.vocab_size), target=label)
        label_mask = label_mask.view(-1)
        num_predict = torch.sum(label_mask)  # 一共有多少是需要预测的
        real_loss = torch.sum(whole_seq_loss * label_mask)  # 总序列真实需要的损失
        predict = torch.argmax(logits, dim=-1).view(-1)  # (bsz * seq_len,)
        num_positive_predict = torch.sum((predict == label) * label_mask)
        return real_loss / num_predict.item(), num_predict.item(), num_positive_predict.item()


class TrainingFramework(object):
    def __init__(self, args):
        self.args = args
        self.device = "cuda:%d" % args.gpu_id

        # 模型
        config = json.load(open(os.path.join(
            project_root_path, f"configs/gcdf_zn_config.json"), "r", encoding="UTF-8")
        )
        self.rpe_model_saved_file_name = os.path.join(
            project_root_path, "ModelStorage/pytorch_model.bin"
        )

        self.model = MLMPretrainModel(config=config, max_seq_len=args.max_seq_len)
        # 加载已经训练的参数
        self.model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"), strict=True)

        # 训练和验证数据集build
        tokenizer = BertTokenizer(vocab_file="vocab.txt")
        train_dataset = PretrainDataset(tokenizer=tokenizer, is_eval=False, max_seq_len=args.max_seq_len)
        dev_dataset = PretrainDataset(tokenizer=tokenizer, is_eval=True, max_seq_len=args.max_seq_len)
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

    def train(self):
        steps = 0
        best_ppl = 99999
        for epoch in range(self.args.num_train_epochs):
            for item in self.train_loader:
                self.optimizer.zero_grad()
                input_ids, input_mask, input_seg, label, label_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["label"], item["label_mask"]
                input_ids = torch.chunk(input_ids, chunks=self.args.batch_chunk, dim=0)
                input_mask = torch.chunk(input_mask, chunks=self.args.batch_chunk, dim=0)
                input_seg = torch.chunk(input_seg, chunks=self.args.batch_chunk, dim=0)
                label = torch.chunk(label, chunks=self.args.batch_chunk, dim=0)
                label_mask = torch.chunk(label_mask, chunks=self.args.batch_chunk, dim=0)
                loss, num_predict, num_pos_predict = 0, 0, 0
                for chunk in range(len(input_ids)):
                    s_loss, s_num_predict, s_num_pos_predict = self.model(
                        input_ids=input_ids[chunk].to(self.device), input_mask=input_mask[chunk].to(self.device),
                        input_seg=input_seg[chunk].to(self.device), label=label[chunk].to(self.device),
                        label_mask=label_mask[chunk].to(self.device)
                    )
                    s_loss = s_loss / len(input_ids)
                    s_loss.backward()
                    loss += s_loss.item()
                    if s_num_predict == 0:
                        num_predict += 1
                    else:
                        num_predict += s_num_predict
                    num_pos_predict += s_num_pos_predict
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                if steps % self.args.print_interval == 0:
                    print(f"{str(datetime.datetime.now())}, 训练步骤, Steps: {steps}, PPL: {np.exp(loss)}, ACC: {num_pos_predict / num_predict}")
                if steps % self.args.eval_interval == 0:
                    ppl, acc = self.eval()
                    print("===============================")
                    print(f"{str(datetime.datetime.now())}, 验证步骤, PPL: {ppl}, ACC: {acc}")
                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save(self.model.state_dict(), f=self.rpe_model_saved_file_name)
                        print("***验证指标提升,预训练模型已经保存在指定目录***")
                    print("===============================")
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
                sum_cnt += cnt
                sum_loss += cnt * loss.item()
                sum_pos_cnt += pos_cnt
        self.model.train()
        return np.exp(sum_loss / sum_cnt), sum_pos_cnt / sum_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恢复GCDF_ZN_BERT任务层参数训练")
    parser.add_argument("--num_workers", type=int, default=0, help="异步数据加载进程数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")  # 待进行压测
    parser.add_argument("--batch_chunk", type=int, default=2, help="梯度累计数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW使用的权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-9, help="最小的学习率")
    parser.add_argument("--warm_up_rate", type=float, default=0.1, help="热启动过程占整个训练过程的比例")
    parser.add_argument("--clip_norm", type=float, default=0.25, help="梯度裁剪最大范数")
    parser.add_argument("--gpu_id", type=int, default=0, help="占用那个GPU显卡")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练epoch数")
    parser.add_argument("--eval_interval", type=int, default=10000, help="模型验证一次间隔")
    parser.add_argument("--print_interval", type=int, default=1000, help="损失打印间隔")

    args = parser.parse_args()
    print("创建预训练对象")
    obj = TrainingFramework(args)
    print("预训练对象创建完成, 执行预训练过程")
    obj.train()
    print("预训练任务完成, 开始新的旅程吧-_-*_*~_~^_^")


