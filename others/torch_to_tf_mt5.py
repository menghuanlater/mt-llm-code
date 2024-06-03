# -*- coding: UTF-8 -*-
"""
@File   : tf_graph.py.py
@Author : quanlin03
@Date   : 2023/2/23 19:45
@Usage  : MT5-small纠错模型pytorch转tf静态图上线 ==> 离线评估发现beam_search加入并没有效果提升, 在召回能力上甚至有略微的小下降, 耗时却
          增加了50%左右, 因此本版本暂不实现beam_search的功能, 如果实现, 可参照Desktop/Point_generator进行相对应的实现, 不需要完全照搬huggingface的

          TODO: 埋点记录
          1. MT5使用的是Root Mean Square Layer Norm, 不能用已经实现的layer_norm
          2. MT5的Attention的QKVO + FFN的bias都取消了
          3. encoder、decoder共享embedding_matrix, output不共享, 由于模型导出的时候这几个tensor都有
          4. MT5的layer norm是提前LN的
          5. T5的RPE只有Encoder和Decoder的第一层有, 并且属于bias项, 共享传递给其他层, 但需要注意的是其采用的是bucket的方式, 近距离精细化, 远距离粗糙化,
          获得bias的方法建议参照T5官方代码进行实现, 比较复杂
          6. Decoder的首字start以<pad>为输入
          7. T5的词表可通过T5Tokenizer进行加载子词模型之后进行导出
          8. T5的attention是没有scaled, 与标准transformer不一样

"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.framework import graph_io, graph_util
import pickle
import datetime
import math
import numpy as np
from tqdm.auto import tqdm

# 如果要使用GPU, 下面的设置必不可少
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True


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
    def __init__(self, vocab_file="t5-base-vocab.pkl"):
        self.id_to_word = dict()
        with open(vocab_file, "rb") as f:
            self.word_to_id = pickle.load(f)
        for word in self.word_to_id.keys():
            self.id_to_word[self.word_to_id[word]] = word

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.word_to_id.keys():
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.word_to_id["<unk>"])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            word = self.id_to_word[id]
            if word == "</s>":
                break
            elif word not in ["<pad>", "<unk>"]:
                tokens.append(word)
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
    max_len = min(len(asr_tokens), len(real_tokens), config["max_enc_len"])
    asr_tokens, real_tokens = asr_tokens[:max_len], real_tokens[:max_len]
    machine_tokens = [i for i in revised_machine_sentence]
    pre_customer_tokens = [i for i in pre_customer_sentences]
    input_ids = tokenizer.convert_tokens_to_ids(
        pre_customer_tokens + ["</s>"] + machine_tokens + ["</s>"] + asr_tokens)
    if len(input_ids) > config["max_enc_len"]:
        input_ids = input_ids[len(input_ids) - config["max_enc_len"]:]
    input_mask = [1] * len(input_ids)
    extra = config["max_enc_len"] - len(input_ids)
    if extra > 0:
        input_ids += [0] * extra
        input_mask += [0] * extra
    return {
        "input_ids": input_ids, "input_mask": input_mask, "asr_text": asr_text, "real_text": real_text
    }


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


# 根据T5的compute_bias和_relative_position_bucket函数整合而得
# is_bidirectional表示的是encoder还是decoder, 在decoder部分, query_length将强制变成1, 对应的需要做一些翻转, 因为我们启动了use_cache的解码方式降低推理时延
def get_relative_position_bucket(query_length, key_length, is_bidirectional, num_buckets, max_distance):
    context_position = tf.range(start=0, limit=query_length, dtype=tf.int32)[:, None]
    relative_buckets = 0
    if not is_bidirectional:  # 说明是解码状态, 此时的query_length一定为1, 设置一个assert
        assert query_length == 1
        memory_position = tf.range(start=1 - key_length, limit=1, delta=1, dtype=tf.int32)
        relative_position = memory_position - context_position  # (1, key_length)
        relative_position = -tf.minimum(relative_position, tf.zeros_like(relative_position))
    else:
        memory_position = tf.range(start=0, limit=key_length, dtype=tf.int32)[None, :]
        relative_position = memory_position - context_position  # (query_length, key_length)
        num_buckets //= 2
        relative_buckets += tf.cast(relative_position > 0, dtype=tf.int32) * num_buckets
        relative_position = tf.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        tf.log(tf.cast(relative_position, dtype=tf.float32) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    )
    relative_position_if_large = tf.cast(relative_position_if_large, dtype=tf.int32)
    relative_position_if_large = tf.minimum(relative_position_if_large, tf.ones_like(relative_position_if_large) * (num_buckets - 1))
    relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


# T5专用的layer_norm
def mt5_layer_norm(hidden_states, pretrain_weight_key):
    ln_weight = tf.get_variable(
        "layer_norm", initializer=np.array(np_weights_map[pretrain_weight_key]),
        trainable=False, dtype=tf.float32
    )
    variance = tf.reduce_mean(tf.pow(hidden_states, 2), axis=-1, keepdims=True)
    hidden_states = hidden_states * tf.rsqrt(variance + config["eps"])
    normed_hidden_states = ln_weight * hidden_states
    return normed_hidden_states


# T5的Attention --> 聚合Encoder的双向attention, Decoder因果attention和cross-attention, 但不包括ln和residual部分 -> 主要是为了绑参数方便
# attn_mode从下面三个值中取 ["encoder", "decoder-causal", "decoder-cross"]
# 返回hidden_states, 同时返回四个past, 按照字典的形式
def mt5_attention(inp, key_value_length, pretrain_weight_prefix, attn_mode, r_bias=None, attn_mask=None,
                  past_cross_attn_key_states=None, past_cross_attn_value_states=None,
                  past_causal_attn_key_states=None, past_causal_attn_value_states=None,
                  encoder_representations=None):
    assert attn_mode in ["encoder", "decoder-causal", "decoder-cross"]
    bsz, query_length = tf.shape(inp)[0], tf.shape(inp)[1]
    query_states = fully_connected(
        scope="query_net", inputs=inp, num_outputs=config["num_heads"] * config["head_dim"],
        trainable=False, activation_fn=None, biases_initializer=None,
        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"{pretrain_weight_prefix}.q.weight"]))
    )
    if attn_mode == "decoder-cross":
        if past_cross_attn_key_states is not None:
            key_states = past_cross_attn_key_states
        else:
            key_states = fully_connected(
                scope="key_net", inputs=encoder_representations, num_outputs=config["num_heads"] * config["head_dim"],
                trainable=False, activation_fn=None, biases_initializer=None,
                weights_initializer=tf.constant_initializer(
                    np.transpose(np_weights_map[f"{pretrain_weight_prefix}.k.weight"]))
            )
        past_cross_attn_key_states = key_states
        if past_cross_attn_value_states is not None:
            value_states = past_cross_attn_value_states
        else:
            value_states = fully_connected(
                scope="value_net", inputs=encoder_representations, num_outputs=config["num_heads"] * config["head_dim"],
                trainable=False, activation_fn=None, biases_initializer=None,
                weights_initializer=tf.constant_initializer(np.transpose(np_weights_map[f"{pretrain_weight_prefix}.v.weight"]))
            )
        past_cross_attn_value_states = value_states
    else:
        key_states = fully_connected(
            scope="key_net", inputs=inp, num_outputs=config["num_heads"] * config["head_dim"],
            trainable=False, activation_fn=None, biases_initializer=None,
            weights_initializer=tf.constant_initializer(
                np.transpose(np_weights_map[f"{pretrain_weight_prefix}.k.weight"]))
        )
        value_states = fully_connected(
            scope="value_net", inputs=inp, num_outputs=config["num_heads"] * config["head_dim"],
            trainable=False, activation_fn=None, biases_initializer=None,
            weights_initializer=tf.constant_initializer(
                np.transpose(np_weights_map[f"{pretrain_weight_prefix}.v.weight"]))
        )

    if attn_mode == "decoder-causal":
        if past_causal_attn_key_states is not None:
            key_states = tf.concat([past_causal_attn_key_states, key_states], axis=1)
        past_causal_attn_key_states = key_states
        if past_causal_attn_value_states is not None:
            value_states = tf.concat([past_causal_attn_value_states, value_states], axis=1)
        past_causal_attn_value_states = value_states

    w_q = tf.reshape(query_states, [bsz, query_length, config["num_heads"], config["head_dim"]])
    w_k = tf.reshape(key_states, [bsz, key_value_length, config["num_heads"], config["head_dim"]])
    w_v = tf.reshape(value_states, [bsz, key_value_length, config["num_heads"], config["head_dim"]])
    attn_score = tf.einsum("bqnd,bknd->bqkn", w_q, w_k)
    if r_bias is not None:
        attn_score += r_bias
    if attn_mask is not None:
        attn_score += (1.0 - attn_mask) * (-1e30)
    attn_prob = tf.nn.softmax(attn_score, axis=2)
    attn_vec = tf.einsum("bqkn,bknd->bqnd", attn_prob, w_v)
    attn_vec = tf.reshape(attn_vec, [bsz, query_length, config["num_heads"] * config["head_dim"]])
    attn_output = fully_connected(
        scope="output_net", inputs=attn_vec, num_outputs=config["hidden_dim"],
        trainable=False, activation_fn=None, biases_initializer=None,
        weights_initializer=tf.constant_initializer(
            np.transpose(np_weights_map[f"{pretrain_weight_prefix}.o.weight"]))
    )
    return {
        "attn_output": attn_output, "past_causal_attn_key_states": past_causal_attn_key_states,
        "past_causal_attn_value_states": past_causal_attn_value_states,
        "past_cross_attn_key_states": past_cross_attn_key_states,
        "past_cross_attn_value_states": past_cross_attn_value_states
    }


# T5的FFN层, 包含完整的ln和residual部分
def ffn_layer(inp, pretrain_weight_prefix):
    # 第一步
    normed_hidden_states = mt5_layer_norm(inp, f"{pretrain_weight_prefix}.layer_norm.weight")
    # 需要排查激活函数是否正确
    hidden_gelu = fully_connected(
        scope="up_0", inputs=normed_hidden_states, num_outputs=config["ffn_size"],
        trainable=False, activation_fn=gelu, biases_initializer=None,
        weights_initializer=tf.constant_initializer(
            np.transpose(np_weights_map[f"{pretrain_weight_prefix}.DenseReluDense.wi_0.weight"]))
    )
    hidden_linear = fully_connected(
        scope="up_1", inputs=normed_hidden_states, num_outputs=config["ffn_size"],
        trainable=False, activation_fn=None, biases_initializer=None,
        weights_initializer=tf.constant_initializer(
            np.transpose(np_weights_map[f"{pretrain_weight_prefix}.DenseReluDense.wi_1.weight"]))
    )
    hidden_up = hidden_linear * hidden_gelu
    ffn_out = fully_connected(
        scope="down", inputs=hidden_up, num_outputs=config["hidden_dim"],
        trainable=False, activation_fn=None, biases_initializer=None,
        weights_initializer=tf.constant_initializer(
            np.transpose(np_weights_map[f"{pretrain_weight_prefix}.DenseReluDense.wo.weight"]))
    )
    # 残差连接
    hidden_states = inp + ffn_out
    return hidden_states


def build_graph(enc_input_ids, enc_input_mask):
    """
    :param enc_input_ids: (bsz, seq_len)
    :param enc_input_mask: (bsz, seq_len)
    :return: decoder_output_ids
    """
    bsz, enc_seq_len = tf.shape(enc_input_ids)[0], tf.shape(enc_input_mask)[1]
    # Encoder部分
    with tf.variable_scope("Encoder"):
        enc_attn_mask = tf.cast(tf.tile(enc_input_mask[:, None, :], multiples=[1, enc_seq_len, 1]), dtype=tf.float32)[..., None]  # (bsz, q_len, k_len, 1)
        token_embedding_matrix = tf.get_variable(
            "token_embeddings", initializer=np.array(np_weights_map["s2s_model.encoder.embed_tokens.weight"]),
            trainable=False, dtype=tf.float32
        )
        token_emb = tf.nn.embedding_lookup(token_embedding_matrix, enc_input_ids)
        # 提前获得相对位置编码偏置矩阵, 因为后续所有的Encoder层都会共享使用这个位置偏置信息
        relative_position_embedding_matrix = tf.get_variable(
            "relative_position_embeddings", initializer=np.array(
                np_weights_map["s2s_model.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
            ), trainable=False, dtype=tf.float32
        )  # (num_buckets, num_heads)
        relative_position_bucket = get_relative_position_bucket(
            enc_seq_len, enc_seq_len, True, config["relative_attention_num_buckets"], config["relative_attention_max_distance"]
        )
        relative_bias = tf.nn.embedding_lookup(relative_position_embedding_matrix, relative_position_bucket)[None, ...]  # (1, q_len, k_len, num_heads)
        hidden_states = token_emb
        for i in range(config["num_encoder_layers"]):
            with tf.variable_scope(f"block-{i}"):
                # 第一大子层, 注意力层
                with tf.variable_scope("self-attention-layer"):
                    # 第一步: pre-ln
                    normed_hidden_states = mt5_layer_norm(hidden_states, f"s2s_model.encoder.block.{i}.layer.0.layer_norm.weight")
                    # 第二步: qkv映射
                    attn_output = mt5_attention(
                        inp=normed_hidden_states, key_value_length=enc_seq_len, attn_mode="encoder",
                        pretrain_weight_prefix=f"s2s_model.encoder.block.{i}.layer.0.SelfAttention",
                        r_bias=relative_bias, attn_mask=enc_attn_mask
                    )["attn_output"]
                    # 残差连接
                    hidden_states = hidden_states + attn_output
                # 第二大子层, FFN层
                with tf.variable_scope("ffn-layer"):
                    hidden_states = ffn_layer(hidden_states, pretrain_weight_prefix=f"s2s_model.encoder.block.{i}.layer.1")
        # Encoder最后一层的ln
        with tf.variable_scope("final-ln"):
            encoder_final_representations = mt5_layer_norm(hidden_states, f"s2s_model.encoder.final_layer_norm.weight")  # (bsz, enc_seq_len, )
    # Decoder部分, 最复杂的部分, 需要考虑cache, 分为causal-self-attention、cross-attention
    past_cross_attn_key_states, past_cross_attn_value_states = [None for _ in range(config["num_decoder_layers"])], \
        [None for _ in range(config["num_decoder_layers"])]
    past_causal_attn_key_states, past_causal_attn_value_states = [None for _ in range(config["num_decoder_layers"])], \
        [None for _ in range(config["num_decoder_layers"])]
    decoder_input_ids = tf.zeros(shape=(bsz, 1), dtype=tf.int32)  # <pad>作为起始解码状态位
    dec2enc_attn_mask = tf.cast(enc_input_mask[:, None, :], dtype=tf.float32)[..., None]  # (bsz, 1, k_len, 1)
    decoder_predict_step_by_step = []  # 每一时刻步的解码结果 -> List[Tensor(bsz, 1)]
    # 标量, 全部填充为</s>的id, 判断解码是否解到了结束符号
    eos_sentry = tf.ones(shape=(bsz, 1), dtype=tf.int32)
    # 是否已经解码到</s>的标志位
    arrive_eos_flag = tf.cast(tf.zeros_like(eos_sentry), dtype=tf.bool)
    with tf.variable_scope("Decoder"):
        token_embedding_matrix = tf.get_variable(
            "token_embeddings", initializer=np.array(np_weights_map["s2s_model.decoder.embed_tokens.weight"]),
            trainable=False, dtype=tf.float32
        )
        relative_position_embedding_matrix = tf.get_variable(
            "relative_position_embeddings", initializer=np.array(
                np_weights_map["s2s_model.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
            ), trainable=False, dtype=tf.float32
        )  # (num_buckets, num_heads)

        def step_decode(l, decoder_input_ids):
            # 获取当前解码时间步的相对位置编码偏置
            relative_position_bucket = get_relative_position_bucket(
                1, l + 1, False, config["relative_attention_num_buckets"],
                config["relative_attention_max_distance"]
            )
            relative_bias = tf.nn.embedding_lookup(relative_position_embedding_matrix, relative_position_bucket)[
                None, ...]  # (1, 1, l + 1, num_heads)
            hidden_states = tf.nn.embedding_lookup(token_embedding_matrix, decoder_input_ids)
            for i in range(config["num_decoder_layers"]):
                with tf.variable_scope(f"block-{i}", reuse=tf.AUTO_REUSE):
                    # 第一大子层, causal-attention层
                    with tf.variable_scope("causal-self-attention-layer"):
                        # 第一步, pre-ln
                        normed_hidden_states = mt5_layer_norm(hidden_states, f"s2s_model.decoder.block.{i}.layer.0.layer_norm.weight")
                        # 第二步, qkv映射
                        temp = mt5_attention(
                            inp=normed_hidden_states, key_value_length=l + 1, attn_mode="decoder-causal",
                            r_bias=relative_bias,
                            pretrain_weight_prefix=f"s2s_model.decoder.block.{i}.layer.0.SelfAttention",
                            past_causal_attn_key_states=past_causal_attn_key_states[i],
                            past_causal_attn_value_states=past_causal_attn_value_states[i]
                        )
                        attn_output = temp["attn_output"]
                        past_causal_attn_key_states[i] = temp["past_causal_attn_key_states"]
                        past_causal_attn_value_states[i] = temp["past_causal_attn_value_states"]
                        # 残差连接
                        hidden_states = attn_output + hidden_states
                    # 第二大子层, cross-attention层
                    with tf.variable_scope("cross-attention-layer"):
                        # 第一步, pre-ln
                        normed_hidden_states = mt5_layer_norm(hidden_states,
                                                              f"s2s_model.decoder.block.{i}.layer.1.layer_norm.weight")
                        # 第二步, qkv映射
                        temp = mt5_attention(
                            inp=normed_hidden_states, key_value_length=enc_seq_len, attn_mode="decoder-cross",
                            pretrain_weight_prefix=f"s2s_model.decoder.block.{i}.layer.1.EncDecAttention",
                            past_cross_attn_key_states=past_cross_attn_key_states[i],
                            past_cross_attn_value_states=past_cross_attn_value_states[i],
                            encoder_representations=encoder_final_representations,
                            attn_mask=dec2enc_attn_mask
                        )
                        attn_output = temp["attn_output"]
                        past_cross_attn_key_states[i] = temp["past_cross_attn_key_states"]
                        past_cross_attn_value_states[i] = temp["past_cross_attn_value_states"]
                        # 残差连接
                        hidden_states = attn_output + hidden_states
                    # 第三大子层, FFN层
                    with tf.variable_scope("ffn-layer"):
                        hidden_states = ffn_layer(hidden_states,
                                                  pretrain_weight_prefix=f"s2s_model.decoder.block.{i}.layer.2")
            # 最后的LN层
            with tf.variable_scope("final-ln", reuse=tf.AUTO_REUSE):
                decoder_final_representations = mt5_layer_norm(hidden_states,
                                                               f"s2s_model.decoder.final_layer_norm.weight")
            # 解码预测
            with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
                logits = fully_connected(
                    scope="vocab_decoder", inputs=decoder_final_representations, num_outputs=config["vocab_size"],
                    trainable=False, activation_fn=None, biases_initializer=None,
                    weights_initializer=tf.constant_initializer(
                        np.transpose(np_weights_map["s2s_model.lm_head.weight"]))
                )
                single_step_decode_result = tf.argmax(logits, axis=-1, output_type=tf.int32)
            return single_step_decode_result

        for l in range(config["max_dec_len"]):
            # 先判断是否完全满足条件了
            arrive_eos_flag = tf.where(arrive_eos_flag, arrive_eos_flag, tf.equal(decoder_input_ids, eos_sentry))
            single_step_decode_result = tf.cond(
                tf.reduce_sum(tf.cast(arrive_eos_flag, dtype=tf.int32)) >= bsz,
                lambda: tf.zeros_like(decoder_input_ids), lambda: step_decode(l, decoder_input_ids)
            )
            decoder_predict_step_by_step.append(single_step_decode_result)
            decoder_input_ids = single_step_decode_result
    s2s_out = tf.cast(tf.concat(decoder_predict_step_by_step, axis=-1), dtype=tf.int64, name="predict")

    return s2s_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Pytorch模型导出TF静态图测试 --> ASRV5纠错上线生成式模型")
    parser.add_argument("--export_tf_model", action="store_true", default=False)
    args = parser.parse_args()

    if args.export_tf_model:
        batch_size = 1
    else:
        batch_size = 32

    config = {
        "vocab_size": 60108, "num_encoder_layers": 8, "num_decoder_layers": 8,
        "hidden_dim": 512, "num_heads": 6, "head_dim": 64, "ffn_size": 1024, "eps": 1e-6,
        "relative_attention_max_distance": 128, "relative_attention_num_buckets": 32,
        "max_enc_len": 128, "max_dec_len": 64
    }
    # 加载权重参数, 为了防止torch和tensorflow环境冲突, 这里是预先在torch环境下读取参数后直接导成二进制pickle文件
    with open("np_weights.pkl", "rb") as f:
        np_weights_map = pickle.load(f)

    tokenizer = Tokenizer()

    # 先用batch为32去测试下计算图是否正确, 测试真实性能时batch_size设置为1
    encoder_input_ids_holder = tf.placeholder(name="encoder_input_ids", dtype=tf.int32, shape=[batch_size, config["max_enc_len"]])
    encoder_input_mask_holder = tf.placeholder(name="encoder_input_mask", dtype=tf.int32, shape=[batch_size, config["max_enc_len"]])
    predict = build_graph(encoder_input_ids_holder, encoder_input_mask_holder)
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    with open("../data/chaotic_finetune_data.pkl", "rb") as f:
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

    first_flag = True and args.export_tf_model
    log_info = []
    if len(valid_data) % batch_size == 0:
        extra = 0
    else:
        extra = batch_size - (len(valid_data) % batch_size)
        valid_data = valid_data + valid_data[:extra]
    for k in tqdm(range(len(valid_data) // batch_size)):
        input_ids, input_mask, asr_text, real_text = [], [], [], []
        batch_items = []
        for p in range(batch_size):
            item = get_input_features(valid_data[batch_size * k + p])
            input_ids.append(item["input_ids"])
            input_mask.append(item["input_mask"])
            asr_text.append(item["asr_text"])
            real_text.append(item["real_text"])
            batch_items.append(item)
        pred = sess.run(predict, feed_dict={
            encoder_input_ids_holder: input_ids, encoder_input_mask_holder: input_mask
        })
        iter_len = batch_size
        if extra > 0 and k == ((len(valid_data) // batch_size) - 1):
            iter_len -= extra
        for i in range(iter_len):
            t_asr_text, t_real_text, t_pred = asr_text[i], real_text[i], "".join(tokenizer.convert_ids_to_tokens(pred[i]))
            t_pred = t_pred[:len(t_asr_text)]
            log_info.append(f"{t_real_text}\t{t_pred}")
            if t_asr_text == t_real_text:
                sentence_y_label.append(1)
            else:
                sentence_y_label.append(0)
            if t_real_text == t_pred:
                sentence_pred_label.append(1)
                if first_flag:
                    if t_real_text != t_asr_text:
                        first_flag = False
                        print(batch_items[i])
                        print(list(pred[i]))
            else:
                sentence_pred_label.append(0)
            length = min(len(t_asr_text), len(t_real_text), len(t_pred))
            t_asr_text, t_real_text, t_pred = t_asr_text[:length], t_real_text[:length], t_pred[:length]
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

    if args.export_tf_model:
        # 最后一步, 保存为saved_model格式, 修改signature为"test_signature"
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["predict"]
        )
        graph_io.write_graph(constant_graph, "./", "model.pb", as_text=False)

        sess.close()

        with open("model.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(config=tf_config)
            tf.import_graph_def(graph_def, name='')
            init = tf.global_variables_initializer()
            sess.run(init)
            inputs = {
                "encoder_input_ids:0": tf.saved_model.utils.build_tensor_info(encoder_input_ids_holder),
                "encoder_input_mask:0": tf.saved_model.utils.build_tensor_info(encoder_input_mask_holder)
            }
            outputs = {"predict:0": tf.saved_model.utils.build_tensor_info(predict)}

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
        print("TF模型导出完毕..., 欢迎使用~_~")
    else:
        with open("log.txt", "w", encoding="UTF-8") as f:
            f.write("真实结果\t预测结果\n")
            for info in log_info:
                f.write(f"{info}\n")
