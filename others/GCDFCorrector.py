# -*- coding: UTF-8 -*-
"""
@File   : GCDFCorrector.py
@Author : quanlin03
@Date   : 2022/9/6 14:36
@Usage  : 使用GCDF-RPE-BERT执行纠错
"""
import torch
from random import shuffle
import pickle
import numpy as np
from LRSchedule import WarmUpLinearDecay
from transformers import BertTokenizer
from rpe_bert import RPEBertModel
from torch import optim
from torch.utils.data import Dataset, DataLoader
import datetime
import json
from collections import OrderedDict


class UnifiedDataset(Dataset):
    # 符合最终建模范式
    def __init__(self, data: list, is_eval: bool, is_pretrain: bool):
        super(UnifiedDataset, self).__init__()
        self.data = data
        self.max_len = 128
        self.is_eval = is_eval
        self.is_pretrain = is_pretrain

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        训练时: input_ids, input_mask, correction_label, dc_mask
        验证时: input_ids, input_mask, asr_text, real_text, start, end
        """
        item = dict(self.data[index])
        asr_text, real_text, machine_sentence, pre_customer_sentences = \
            item["asr_text"].replace(" ", ""), item["real_text"].replace(" ", ""), item["machine_sentence"], item["pre_customer_sentences"]
        if type(pre_customer_sentences) != str:
            pre_customer_sentences = "|".join(pre_customer_sentences)
        # 使用关键词修正当前的机器人话术
        revised_machine_sentence_list = []
        left_ptr, right_ptr = 0, 1
        cur_match_prefix = ""
        while right_ptr <= len(machine_sentence):
            if machine_sentence[left_ptr:right_ptr] not in machine_meaningful_prefix:
                if len(cur_match_prefix) > 0:
                    if cur_match_prefix in machine_meaningful_vocab:
                        revised_machine_sentence_list.append(cur_match_prefix)
                    cur_match_prefix = ""
                    left_ptr = right_ptr - 1
                else:
                    left_ptr += 1
                    right_ptr += 1
            else:
                cur_match_prefix = machine_sentence[left_ptr:right_ptr]
                right_ptr += 1
        if len(cur_match_prefix) > 0:
            if cur_match_prefix in machine_meaningful_vocab:
                revised_machine_sentence_list.append(cur_match_prefix)
        revised_machine_sentence = "-".join(revised_machine_sentence_list)
        # 修正结束标志
        asr_tokens = [i for i in asr_text]
        real_tokens = [i for i in real_text]
        max_len = min(len(asr_tokens), len(real_tokens), self.max_len)
        asr_tokens, real_tokens = asr_tokens[:max_len], real_tokens[:max_len]
        detection_label = []
        for i in range(len(asr_tokens)):
            if asr_tokens[i] == real_tokens[i]:
                detection_label.append(1)
            else:
                detection_label.append(0)
        dc_mask = [1] * len(asr_tokens)
        machine_tokens = [i for i in revised_machine_sentence]
        pre_customer_tokens = [i for i in pre_customer_sentences]
        input_ids = tokenizer.convert_tokens_to_ids(pre_customer_tokens + ["[SEP]"] + machine_tokens + ["[SEP]"] + asr_tokens)
        input_seg = [1] * (len(pre_customer_tokens) + len(machine_tokens) + 2) + [0] * len(asr_tokens)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[len(input_ids) - self.max_len:]
            input_seg = input_seg[len(input_seg) - self.max_len:]
        extra = len(input_ids) - len(asr_tokens)
        detection_label = [0] * extra + detection_label
        correction_label = [0] * extra + tokenizer.convert_tokens_to_ids(real_tokens)
        dc_mask = [0] * extra + dc_mask
        start, end = len(input_ids) - len(asr_tokens), len(input_ids) - 1
        input_mask = [1] * len(input_ids)
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            input_mask += [0] * extra
            input_seg += [0] * extra
            correction_label += [0] * extra
            dc_mask += [0] * extra
            detection_label += [0] * extra
        if self.is_eval:
            return {
                "input_ids": input_ids, "input_mask": input_mask, "asr_text": asr_text, "real_text": real_text,
                "start": start, "end": end, "input_seg": input_seg
            }
        else:
            return {
                "input_ids": input_ids, "input_mask": input_mask, "input_seg": input_seg, "detection_label": detection_label,
                "correction_label": correction_label, "dc_mask": dc_mask
            }


def collate_fn(batch):
    input_ids, input_mask, input_seg = [], [], []
    if "asr_text" in batch[0].keys():  # 验证模式
        asr_text, real_text, start, end = [], [], [], []
        for item in batch:
            input_ids.append(item["input_ids"])
            input_mask.append(item["input_mask"])
            input_seg.append(item["input_seg"])
            asr_text.append(item["asr_text"])
            real_text.append(item["real_text"])
            start.append(item["start"])
            end.append(item["end"])
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_mask": torch.tensor(input_mask).long(),
            "input_seg": torch.tensor(input_seg).long(),
            "asr_text": asr_text, "real_text": real_text, "start": start, "end": end
        }
    else:
        detection_label, correction_label, dc_mask = [], [], []
        for item in batch:
            input_ids.append(item["input_ids"])
            input_mask.append(item["input_mask"])
            input_seg.append(item["input_seg"])
            detection_label.append(item["detection_label"])
            correction_label.append(item["correction_label"])
            dc_mask.append(item["dc_mask"])
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_mask": torch.tensor(input_mask).long(),
            "input_seg": torch.tensor(input_seg).long(), "detection_label": torch.tensor(detection_label).long(),
            "correction_label": torch.tensor(correction_label).long(), "dc_mask": torch.tensor(dc_mask).long()
        }


class GCDFCorrector(torch.nn.Module):
    def __init__(self, bert_config: dict):
        """
        :param bert_config: bert的相关配置项, 需要以下参数
        hidden_dim, vocab_size, num_positions, dropout_norm, layer_norm_eps, n_layers, dropout_attn, n_heads, ffn_size, mask_token_id
        """
        super(GCDFCorrector, self).__init__()
        self.bert_config = bert_config

        self.bert_encoder = RPEBertModel(config=bert_config, rel_method="gcdf", max_seq_len=128)
        self.rep_transform = torch.nn.Linear(in_features=bert_config["hidden_dim"], out_features=bert_config["hidden_dim"])
        self.rep_layer_norm = torch.nn.LayerNorm(bert_config["hidden_dim"], eps=bert_config["layer_norm_eps"])
        self.decoder = torch.nn.Linear(in_features=bert_config["hidden_dim"], out_features=bert_config["vocab_size"])

        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids, input_mask, input_seg, correction_label=None, dc_mask=None, detection_label=None):
        encoder_rep = self.bert_encoder(input_ids, input_mask, input_seg)
        correction_rep = self.rep_layer_norm(self.rep_transform(encoder_rep))

        vocab_logits = self.decoder(correction_rep)
        vocab_prob = torch.softmax(vocab_logits, dim=-1)

        if correction_label is not None:  # 训练模式
            # 先计算检测网络的损失
            correction_loss = torch.sum(
                self.loss_function(target=correction_label.view(-1), input=vocab_logits.view(-1, self.bert_config["vocab_size"])) * dc_mask.view(-1)
            ) / torch.sum(dc_mask)
            return correction_loss
        else:
            return torch.argmax(vocab_prob, dim=-1)  # (bsz, seq_len)


class TrainingFramework(object):
    def __init__(self, args, bert_config):
        self.args = args
        self.bert_config = bert_config
        self.device = "cuda:%d" % args.gpu_id

        # 数据集的构建
        with open(args.dataset_path, "rb") as f:
            obj = pickle.load(f)
        shuffle(obj["train_data"])
        train_dataset = UnifiedDataset(data=obj["train_data"], is_eval=False, is_pretrain=args.is_pretrain)
        valid_dataset = UnifiedDataset(data=obj["valid_data"], is_eval=True, is_pretrain=args.is_pretrain)
        self.train_loader = DataLoader(
            dataset=train_dataset, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            dataset=valid_dataset, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size,
            collate_fn=collate_fn
        )

        # 创建模型
        self.model = GCDFCorrector(bert_config=bert_config)
        if args.load_pretrain_weight:
            if args.not_strict_bind:  # 不完全绑定需要先用原始bert初始化, 之后再加载, 防止某些新分支参数完全随机初始化
                self.model.bert_encoder.load_state_dict(self.get_pretrain_weights_state_dict(), strict=False)
                self.model.load_state_dict(
                    torch.load(f"ModelStorage/{args.pretrain_model_name}.pth", map_location="cpu"), strict=False)
            else:
                self.model.load_state_dict(
                    torch.load(f"ModelStorage/{args.pretrain_model_name}.pth", map_location="cpu"), strict=True)
        else:
            # bert部分加载预训练权重
            self.model.bert_encoder.load_state_dict(self.get_pretrain_weights_state_dict(), strict=False)

        # 设置模型的保存名
        if not args.is_pretrain:
            self.model_saved_path = f"ModelStorage/{args.saved_prefix}.pth"
            self.valid_outcome_path = f"valid_outcome/{args.saved_prefix}.json"
        else:
            self.model_saved_path = f"ModelStorage/{args.saved_prefix}_Pretrain.pth"
            self.valid_outcome_path = None

        # 优化器设置
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)
        all_steps = int(args.num_train_epochs * train_dataset.__len__() / args.batch_size)
        self.schedule = WarmUpLinearDecay(
            optimizer=self.optimizer, init_rate=args.lr, warm_up_steps=int(args.warm_up_rate * all_steps),
            decay_steps=int((1.0 - args.warm_up_rate) * all_steps), min_lr_rate=args.min_lr
        )
        self.model.to(self.device)

    @staticmethod
    def get_pretrain_weights_state_dict():
        pretrain_state_dict = torch.load("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/Pretrain_Models/GCDF_Base_ZH/pytorch_model.bin")
        return pretrain_state_dict

    def train(self):
        self.model.train()
        best_metric = 0
        best_all_metrics = None  # 当best_metric更新时, 同样更新其他所有的metric, 融合安置在一个dict中
        steps = 0
        for epoch in range(self.args.num_train_epochs):
            if epoch == self.args.stop_epoch:
                break
            for item in self.train_loader:
                self.optimizer.zero_grad()
                input_ids, input_mask, input_seg, detection_label, correction_label, dc_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["detection_label"], \
                    item["correction_label"], item["dc_mask"]
                loss = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device), detection_label=detection_label.to(self.device),
                    correction_label=correction_label.to(self.device),
                    dc_mask=dc_mask.to(self.device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)
                self.schedule.step()
                steps += 1
                if steps % self.args.print_interval == 0:
                    print("|| [Time:{}] || [Epoch:{}] || [Steps:{}] || [Loss:{:.3f}]".format(
                        datetime.datetime.now(), epoch + 1, steps, loss.item()))
                if self.args.is_interval_valid and (steps % self.args.valid_interval == 0):
                    print("===到达间隔验证点, 执行验证集验证流程===")
                    valid_metrics = self.valid()
                    print(valid_metrics)
                    if valid_metrics["sentence-level-acc"]["now"] > best_metric:
                        best_metric = valid_metrics["sentence-level-acc"]["now"]
                        best_all_metrics = valid_metrics
                        torch.save(self.model.state_dict(), f=self.model_saved_path)
            if not self.args.is_interval_valid:
                print("===一轮Epoch已经训练完成, 执行验证集验证流程===")
                valid_metrics = self.valid()
                print(valid_metrics)
                if valid_metrics["sentence-level-acc"]["now"] > best_metric:
                    best_metric = valid_metrics["sentence-level-acc"]["now"]
                    best_all_metrics = valid_metrics
                    torch.save(self.model.state_dict(), f=self.model_saved_path)

        # 将最佳的验证集表现的各项指标记录在文件内, 方便后续进行多轮平均的计算
        # 如果是预训练状态, 则不需要保存了
        if not self.args.is_pretrain:
            with open(self.valid_outcome_path, "w", encoding="UTF-8") as f:
                json.dump(best_all_metrics, f)

    def valid(self):
        # 切换进入验证状态
        eval_out = {
            "char-level-acc": {"pre": 0, "now": 0}, "char-level-precision": 0, "char-level-recall": 0, "char-level-f1": 0,
            "sentence-level-acc": {"pre": 0, "now": 0}, "sentence-level-precision": 0, "sentence-level-recall": 0,
            "sentence-level-f1": 0
        }
        self.model.eval()
        # 特别注意, 这里的label含义与通常的label不一样
        sentence_y_label, char_y_label = [], []
        sentence_pred_label, char_pred_label = [], []
        with torch.no_grad():
            for item in self.valid_loader:
                input_ids, input_mask, input_seg, asr_text, real_text, start, end = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["asr_text"], item["real_text"], item["start"], item["end"]
                pred = self.model(
                    input_ids=input_ids.to(self.device), input_mask=input_mask.to(self.device),
                    input_seg=input_seg.to(self.device)
                )
                pred = pred.cpu().numpy()

                for i in range(len(asr_text)):
                    t_asr_text, t_real_text, t_start, t_end, t_pred = asr_text[i], real_text[i], start[i], end[i], pred[i]
                    t_pred = "".join(tokenizer.convert_ids_to_tokens(list(t_pred[t_start: t_end + 1])))
                    t_pred = t_pred.replace("[UNK][UNK][UNK][UNK]", "好的好的")
                    t_pred = t_pred.replace("[UNK][UNK][UNK]", "APP")
                    t_pred = t_pred.replace("[UNK][UNK]", "好的")
                    if t_asr_text == t_real_text:
                        sentence_y_label.append(1)
                    else:
                        sentence_y_label.append(0)
                    length = t_end - t_start + 1
                    t_asr_text, t_real_text = t_asr_text[:length], t_real_text[:length]
                    if t_real_text == t_pred:
                        sentence_pred_label.append(1)
                    else:
                        sentence_pred_label.append(0)
                    for j in range(length):
                        if t_asr_text[j] == t_real_text[j]:
                            char_y_label.append(1)
                        else:
                            char_y_label.append(0)
                        if t_pred[j] == t_real_text[j]:
                            char_pred_label.append(1)
                        else:
                            char_pred_label.append(0)
        # 计算各项指标
        # 1. 准确率计算
        eval_out["char-level-acc"]["pre"] = np.sum(char_y_label) / len(char_y_label)
        eval_out["char-level-acc"]["now"] = np.sum(char_pred_label) / len(char_pred_label)
        eval_out["sentence-level-acc"]["pre"] = np.sum(sentence_y_label) / len(sentence_y_label)
        eval_out["sentence-level-acc"]["now"] = np.sum(sentence_pred_label) / len(sentence_pred_label)

        # 2. 计算字级别P/R/F1
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(char_y_label)):
            if char_y_label[i] == 0 and char_pred_label[i] == 1:
                tp += 1
            elif char_y_label[i] == 0 and char_pred_label[i] == 0:
                fn += 1
            elif char_y_label[i] == 1 and char_pred_label[i] == 1:
                tn += 1
            else:
                fp += 1
        eval_out["char-level-precision"] = tp / (tp + fp + 1e-3)
        eval_out["char-level-recall"] = tp / (tp + fn + 1e-3)
        eval_out["char-level-f1"] = (2 * eval_out["char-level-precision"] * eval_out["char-level-recall"]) / (1e-3 + eval_out["char-level-recall"] + eval_out["char-level-precision"])
        # 3. 计算句子级别P/R/F1
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(sentence_y_label)):
            if sentence_y_label[i] == 0 and sentence_pred_label[i] == 1:
                tp += 1
            elif sentence_y_label[i] == 0 and sentence_pred_label[i] == 0:
                fn += 1
            elif sentence_y_label[i] == 1 and sentence_pred_label[i] == 1:
                tn += 1
            else:
                fp += 1
        eval_out["sentence-level-precision"] = tp / (tp + fp + 1e-3)
        eval_out["sentence-level-recall"] = tp / (tp + fn + 1e-3)
        eval_out["sentence-level-f1"] = (2 * eval_out["sentence-level-precision"] * eval_out["sentence-level-recall"]) / (1e-3 + eval_out["sentence-level-recall"] + eval_out["sentence-level-precision"])
        # 回退训练状态
        self.model.train()
        return eval_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--stop_epoch", type=int, default=-1)
    parser.add_argument("--is_interval_valid", action="store_true", default=False, help="是否采用间隔式验证, 默认是一个epoch结束进行一次验证")
    parser.add_argument("--valid_interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-9)
    parser.add_argument("--warm_up_rate", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=float, default=0.25)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--dataset_path", type=str, default="data/finetune_data.pkl")
    parser.add_argument("--saved_prefix", type=str, default="GCDF")

    # 预训练相关
    parser.add_argument("--is_pretrain", action="store_true", default=False, help="预训练标志")

    # 加载已经训练好的权重
    parser.add_argument("--load_pretrain_weight", action="store_true", default=False)
    parser.add_argument("--not_strict_bind", action="store_true", default=False, help="绑定预训练参数的时候是否需要严格绑定")
    parser.add_argument("--pretrain_model_name", type=str, default="")

    args = parser.parse_args()

    bert_config = {
        "n_layers": 12, "n_heads": 12, "hidden_dim": 768, "ffn_size": 3072, "initializer_std": 0.02,
        "layer_norm_eps": 1e-12, "max_relative_distance": 255, "dropout_attn": 0.1, "dropout_norm": 0.1,
        "vocab_size": 21128
    }

    machine_meaningful_vocab = "美团、生活费、APP、月付、金融、款项、录音、账单、合同、系统、分期、资格、平台、客服、钱包、罚息、app、" \
                               "高风险、风险、借钱、利息、政策、案件、部门、宽限期、高危、备案、责任、违约、资料、" \
                               "信用、征信、正规、失信、信誉、九点、今晚、今天、晚上、逾期、二次违约、打电话、没人接、" \
                               "没接、查账、核实、核查、取消、登录、通知、升级、延期、撤销、升高、上报、上传、放弃、移交、宽限、审核、" \
                               "关闭、协商、偿还、一次性、流转、结清、处理、周转、承诺、遵守、报备、登记、拖欠、拖着、敷衍、" \
                               "严重、影响、家庭、负担、负面、恶意、家人、朋友、亲戚".split("、")
    machine_meaningful_prefix = set()
    for word in machine_meaningful_vocab:
        for i in range(len(word)):
            machine_meaningful_prefix.add(word[:i + 1])

    tokenizer = BertTokenizer(vocab_file="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/quanlin03/Pretrain_Models/GCDF_Base_ZH/vocab.txt")

    obj = TrainingFramework(args, bert_config)
    print("开始训练模型")
    obj.train()
    print("模型训练完毕")

