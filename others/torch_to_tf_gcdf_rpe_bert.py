# -*- coding: UTF-8 -*-
"""
@File   : tf_graph.py
@Author : quanlin03
@Date   : 2022/10/28 10:16
@Usage  : V3纠错模型导出, 带GCDF-RPE
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, layer_norm
from tensorflow.python.framework import graph_io, graph_util
import pickle
import datetime
import numpy as np
import scipy.stats as st


# 如果要使用GPU, 下面的设置必不可少
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


machine_meaningful_vocab = "美团、生活费、APP、月付、金融、款项、录音、账单、合同、系统、分期、资格、平台、客服、钱包、罚息、app、" \
                               "高风险、风险、借钱、利息、政策、案件、部门、宽限期、高危、备案、责任、违约、资料、自动放弃、找家人、找亲戚、找朋友、" \
                               "信用、征信、正规、失信、信誉、九点、今晚、今天、晚上、逾期、二次违约、打电话、没人接、" \
                               "没接、查账、核实、核查、取消、登录、通知、升级、延期、撤销、升高、上报、上传、放弃、移交、宽限、审核、" \
                               "关闭、协商、偿还、一次性、流转、结清、处理、周转、承诺、遵守、报备、登记、拖欠、拖着、敷衍、" \
                               "严重、影响、家庭、负担、负面、恶意、家人、朋友、亲戚、抛给我们、义务、责任".split("、")
machine_meaningful_prefix = set()
for word in machine_meaningful_vocab:
    for i in range(len(word)):
        machine_meaningful_prefix.add(word[:i + 1])


class Tokenizer(object):
    def __init__(self, vocab_file="vocab.txt"):
        self.id_to_word = dict()
        self.word_to_id = dict()
        with open(vocab_file, "r", encoding="UTF-8") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                self.id_to_word[i] = line
                self.word_to_id[line] = i

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.word_to_id.keys():
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.word_to_id["[UNK]"])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            tokens.append(self.id_to_word[id])
        return tokens


def get_input_features(item):
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
    asr_tokens = [i for i in asr_text]
    real_tokens = [i for i in real_text]
    max_len = min(len(asr_tokens), len(real_tokens), args["max_len"])
    asr_tokens, real_tokens = asr_tokens[:max_len], real_tokens[:max_len]
    machine_tokens = [i for i in revised_machine_sentence]
    pre_customer_tokens = [i for i in pre_customer_sentences]
    input_ids = tokenizer.convert_tokens_to_ids(
        pre_customer_tokens + ["[SEP]"] + machine_tokens + ["[SEP]"] + asr_tokens)
    input_seg = [1] * (len(pre_customer_tokens) + len(machine_tokens) + 2) + [0] * len(asr_tokens)
    if len(input_ids) > args["max_len"]:
        input_ids = input_ids[len(input_ids) - args["max_len"]:]
        input_seg = input_seg[len(input_seg) - args["max_len"]:]
    start, end = len(input_ids) - len(asr_tokens), len(input_ids) - 1
    input_mask = [1] * len(input_ids)
    extra = args["max_len"] - len(input_ids)
    if extra > 0:
        input_ids += [0] * extra
        input_mask += [0] * extra
        input_seg += [0] * extra
    return {
        "input_ids": [input_ids], "input_mask": [input_mask], "asr_text": [asr_text], "real_text": [real_text],
        "start": [start], "end": [end], "input_seg": [input_seg]
    }


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def build_graph(input_ids, input_mask, input_seg):
    input_ids = tf.transpose(input_ids, [1, 0])
    input_mask = tf.transpose(input_mask, [1, 0])
    input_seg = tf.transpose(input_seg, [1, 0])

    attn_mask = tf.tile(input_mask[None, :, :], multiples=[args["max_len"], 1, 1])
    attn_mask = tf.cast(attn_mask, dtype=tf.float32)

    # 相对位置编码所需要的gather_indices
    x = np.zeros(shape=(args["max_len"], args["max_len"]), dtype=int)
    max_relative_distance = 255
    for i in range(args["max_len"]):
        for j in range(args["max_len"]):
            if i - j < -max_relative_distance:  # 距离右边/下文的key过远
                x[i, j] = i * (2 * max_relative_distance + 1)
            elif i - j > max_relative_distance:  # 距离左边/上文的key过远
                x[i, j] = (i + 1) * (2 * max_relative_distance + 1) - 1
            else:
                x[i, j] = i * (2 * max_relative_distance + 1) + max_relative_distance - i + j
    x = np.reshape(x, newshape=(-1,))
    rpe_indices = tf.convert_to_tensor(x, dtype=tf.int32)

    pos_seq = np.arange(start=-max_relative_distance, stop=max_relative_distance + 1, step=1.0)
    x = []
    for i in range(1, bert_config["hidden_dim"] + 1):
        x.append(4 * st.norm.cdf(x=pos_seq, loc=0, scale=bert_config["hidden_dim"] ** (i / bert_config["hidden_dim"])))
    gaussian_matrix = tf.convert_to_tensor(np.array(x).transpose((1, 0)), dtype=tf.float32)

    # 先建立embedding层
    with tf.variable_scope("embedding"):
        token_embeddings = tf.get_variable(
            "token_embeddings", initializer=np.array(np_weights_map["bert_encoder.token_embeddings.weight"]),
            trainable=False, dtype=tf.float32
        )
        segment_embeddings = tf.get_variable(
            "segment_embeddings", initializer=np.array(np_weights_map["bert_encoder.segment_embeddings.weight"]),
            trainable=False, dtype=tf.float32
        )
        token_emb = tf.nn.embedding_lookup(token_embeddings, input_ids)
        seg_emb = tf.nn.embedding_lookup(segment_embeddings, input_seg)
        inp = layer_norm(
            token_emb + seg_emb, trainable=False, gamma_initializer=tf.constant_initializer(np_weights_map["bert_encoder.embedding_layer_norm.weight"]),
            beta_initializer=tf.constant_initializer(np_weights_map["bert_encoder.embedding_layer_norm.bias"]),
            begin_norm_axis=-1
        )

    # 交互层建立
    scale = 1 / ((bert_config["hidden_dim"] // bert_config["n_heads"]) ** 0.5)
    with tf.variable_scope("interaction"):
        for i in range(bert_config["n_layers"]):
            with tf.variable_scope(f"layer_{i}"):
                with tf.variable_scope("attention"):
                    w_head_q = fully_connected(
                        scope="query_net", inputs=inp, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.query_net.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.query_net.bias"])
                    )
                    w_head_k = fully_connected(
                        scope="key_net", inputs=inp, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.key_net.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.key_net.bias"])
                    )
                    w_head_v = fully_connected(
                        scope="value_net", inputs=inp, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.value_net.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.value_net.bias"])
                    )
                    w_head_q = tf.reshape(w_head_q, [args["max_len"], 1, bert_config["n_heads"], bert_config["hidden_dim"] // bert_config["n_heads"]])
                    w_head_k = tf.reshape(w_head_k, [args["max_len"], 1, bert_config["n_heads"], bert_config["hidden_dim"] // bert_config["n_heads"]])
                    w_head_v = tf.reshape(w_head_v, [args["max_len"], 1, bert_config["n_heads"], bert_config["hidden_dim"] // bert_config["n_heads"]])

                    # 相对位置编码部分
                    r_w_bias = tf.convert_to_tensor(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.r_w_bias"])
                    r_r_bias = tf.convert_to_tensor(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.r_r_bias"])
                    rel_pos_key = fully_connected(
                        scope="r_net", inputs=gaussian_matrix, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.rel_pos_encoding.r_net.weight"])),
                        biases_initializer=tf.constant_initializer(np.zeros(shape=(bert_config["hidden_dim"],)))
                    )
                    rel_pos_key = tf.reshape(rel_pos_key, shape=(2 * max_relative_distance + 1, bert_config["n_heads"], bert_config["d_head"]))
                    matmul_query_key = tf.einsum("ibnd,jbnd->ijbn", w_head_q + r_w_bias, w_head_k)
                    matmul_query_pos = tf.einsum("ibnd,rnd->irbn", w_head_q + r_r_bias, rel_pos_key)
                    matmul_query_pos = tf.reshape(matmul_query_pos, shape=(-1, 1, bert_config["n_heads"]))
                    matmul_query_pos = tf.gather(matmul_query_pos, indices=rpe_indices, axis=0)
                    matmul_query_pos = tf.reshape(matmul_query_pos, shape=(args["max_len"], args["max_len"], 1, bert_config["n_heads"]))
                    attn_score = matmul_query_key + matmul_query_pos
                    attn_score = attn_score * scale
                    # 加入mask
                    attn_score = attn_score + (1 - attn_mask[:, :, :, None]) * (-1e30)
                    attn_prob = tf.nn.softmax(attn_score, axis=1)
                    attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
                    attn_vec = tf.reshape(attn_vec, shape=[args["max_len"], 1, bert_config["hidden_dim"]])
                    attn_out = fully_connected(
                        scope="o_net", inputs=attn_vec, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.o_net.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.o_net.bias"])
                    )
                    output = layer_norm(
                        attn_out + inp, trainable=False, gamma_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.layer_norm.weight"]),
                        beta_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.enc_attn.layer_norm.bias"]),
                        begin_norm_axis=-1
                    )
                with tf.variable_scope("ffn_layer"):
                    up = fully_connected(
                        scope="up", inputs=output, num_outputs=bert_config["ffn_size"], trainable=False, activation_fn=gelu,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.CoreNet.0.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.CoreNet.0.bias"])
                    )
                    down = fully_connected(
                        scope="down", inputs=up, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.CoreNet.3.weight"])),
                        biases_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.CoreNet.3.bias"])
                    )
                    inp = layer_norm(
                        down + output, trainable=False, gamma_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.layer_norm.weight"]),
                        beta_initializer=tf.constant_initializer(np_weights_map[f"bert_encoder.layers.{i}.ffn_layer.layer_norm.bias"]),
                        begin_norm_axis=-1
                    )

    # 解码层建立
    with tf.variable_scope("decode"):
        output = fully_connected(
            scope="rep_transform", inputs=inp, num_outputs=bert_config["hidden_dim"], trainable=False, activation_fn=None,
            weights_initializer=tf.constant_initializer(np.transpose(np_weights_map["rep_transform.weight"])),
            biases_initializer=tf.constant_initializer(np_weights_map["rep_transform.bias"])
        )
        output = layer_norm(
            output, trainable=False, gamma_initializer=tf.constant_initializer(np_weights_map[f"rep_layer_norm.weight"]),
            beta_initializer=tf.constant_initializer(np_weights_map[f"rep_layer_norm.bias"]),
            begin_norm_axis=-1
        )
        logits = fully_connected(
            scope="vocab_decoder", inputs=output, num_outputs=bert_config["vocab_size"], trainable=False, activation_fn=None,
            weights_initializer=tf.constant_initializer(np.transpose(np_weights_map["decoder.weight"])),
            biases_initializer=tf.constant_initializer(np_weights_map["decoder.bias"])
        )
        predict = tf.transpose(tf.argmax(logits, axis=-1), [1, 0], name="predict")
        return predict  # 转置, 将batch维度提到第一维度去, 恢复


if __name__ == "__main__":
    args = {
        "max_len": 128
    }
    bert_config = {
        "n_layers": 12, "hidden_dim": 768, "ffn_size": 3072,
        "vocab_size": 21128, "num_positions": 512, "n_heads": 12, "d_head": 64
    }
    # 加载权重参数
    with open("np_weights.pkl", "rb") as f:
        np_weights_map = pickle.load(f)

    tokenizer = Tokenizer()

    input_ids_holder = tf.placeholder(name="input_ids", dtype=tf.int32, shape=[1, args["max_len"]])
    input_mask_holder = tf.placeholder(name="input_mask", dtype=tf.int32, shape=[1, args["max_len"]])
    input_seg_holder = tf.placeholder(name="input_seg", dtype=tf.int32, shape=[1, args["max_len"]])
    predict = build_graph(input_ids_holder, input_mask_holder, input_seg_holder)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    with open("../data/finetune_data.pkl", "rb") as f:
        valid_data = pickle.load(f)["valid_data"]

    print("验证数据量:%d" % len(valid_data))
    print("验证开始时间: %s" % str(datetime.datetime.now()))

    eval_out = {
        "char-level-acc": {"pre": 0, "now": 0}, "char-level-precision": 0, "char-level-recall": 0, "char-level-f1": 0,
        "sentence-level-acc": {"pre": 0, "now": 0}, "sentence-level-precision": 0, "sentence-level-recall": 0,
        "sentence-level-f1": 0
    }

    sentence_y_label, char_y_label = [], []
    sentence_pred_label, char_pred_label = [], []
    all_pred_cases = []

    first_flag = True
    for t in valid_data:
        item = get_input_features(t)
        input_ids, input_mask, input_seg, asr_text, real_text, start, end = \
            item["input_ids"], item["input_mask"], item["input_seg"], item["asr_text"], item["real_text"], item["start"], item["end"]
        pred = sess.run(predict, feed_dict={
            input_ids_holder: input_ids, input_mask_holder: input_mask, input_seg_holder: input_seg
        })

        for i in range(len(asr_text)):
            t_asr_text, t_real_text, t_start, t_end, t_pred = asr_text[i], real_text[i], start[i], end[i], pred[i]
            t_pred = "".join(tokenizer.convert_ids_to_tokens(list(t_pred[t_start: t_end + 1])))
            t_pred = t_pred.replace("[UNK][UNK][UNK][UNK]", "好的好的")
            t_pred = t_pred.replace("[UNK][UNK][UNK]", "APP")
            t_pred = t_pred.replace("[UNK][UNK]", "好的")
            all_pred_cases.append(t_pred)
            if t_asr_text == t_real_text:
                sentence_y_label.append(1)
            else:
                sentence_y_label.append(0)
            length = t_end - t_start + 1
            t_asr_text, t_real_text = t_asr_text[:length], t_real_text[:length]
            if t_real_text == t_pred:
                sentence_pred_label.append(1)
                if first_flag:
                    if t_real_text != t_asr_text:
                        first_flag = False
                        print(item)
                        print(list(pred[0]))
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
    print("验证结束时间: %s" % str(datetime.datetime.now()))
    print("\n")
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
    eval_out["char-level-f1"] = (2 * eval_out["char-level-precision"] * eval_out["char-level-recall"]) / (
                1e-3 + eval_out["char-level-recall"] + eval_out["char-level-precision"])
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
    eval_out["sentence-level-f1"] = (2 * eval_out["sentence-level-precision"] * eval_out["sentence-level-recall"]) / (
                1e-3 + eval_out["sentence-level-recall"] + eval_out["sentence-level-precision"])
    print(eval_out)
    with open("pred.pkl", "wb") as f:
        pickle.dump(all_pred_cases, f)
    # 最后一步, 保存为saved_model格式, 修改signature为"test_signature"
    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["decode/predict"]
    )
    graph_io.write_graph(constant_graph, "./", "model.pb", as_text=False)

    sess.close()

    with open("model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        tf.import_graph_def(graph_def, name='')
        init = tf.global_variables_initializer()
        sess.run(init)
        inputs = {
            "input_ids:0": tf.saved_model.utils.build_tensor_info(input_ids_holder),
            "input_mask:0": tf.saved_model.utils.build_tensor_info(input_mask_holder),
            "input_seg:0": tf.saved_model.utils.build_tensor_info(input_seg_holder)
        }
        outputs = {"decode/predict:0": tf.saved_model.utils.build_tensor_info(predict)}

        output_model = "./saved_model"
        signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        ))

        b = tf.saved_model.builder.SavedModelBuilder(output_model)
        b.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                       signature_def_map={"test_signature": signature})
        b.save()

