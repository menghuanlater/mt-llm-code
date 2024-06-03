# -*- coding: UTF-8 -*-
"""
@File   : tf_graph.py
@Author : quanlin03
@Date   : 2022/8/17 12:45
@Usage  : TBD
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import pickle
import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 如果要使用GPU, 下面的设置必不可少
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

# 加载权重参数
with open("np_weights.pkl", "rb") as f:
    np_weights_map = pickle.load(f)


def preprocess_all_data(data, window_s: int, segment_ms: int, max_machine_duration: int):
    # 根据预设的窗口大小和片段大小整理数据
    """
    :param data: 原始的未处理的数据
    :param window_s: 窗口大小 -> 秒数
    :param segment_ms: 片段大小, 最小的时间建模单元 -> 毫秒 // 需要保证窗口大小真实的时间数值可以被片段大小整除
    :param max_machine_duration

    处理事项:
    1. 过滤掉不需要预测的
    2. 根据训练和验证的标志划分
    3. 每通录音的最大时刻 -> 比较machine和customer的最后时间戳即可
    4. 为每个预测样本生成前序machine的时间序列以及前序的customer时间序列

    final: shuffle训练集, 验证集不动
    """
    def change_to_std_time(init_time):
        if init_time % segment_ms >= (segment_ms // 2):
            return (init_time // segment_ms) + 1
        else:
            return init_time // segment_ms

    window_size = (window_s * 1000) // segment_ms
    valid_data = []
    for conv_item in data:
        if len(conv_item["machine_seq"]) == 0 or len( conv_item["customer_seq"]) == 0:
            continue
        # 第一步, 找到最大时刻
        max_t = max(conv_item["machine_seq"][-1]["end"], conv_item["customer_seq"][-1]["end"])
        if len(conv_item["illegal_customer_seq"]) > 0:
            max_t = max(max_t, conv_item["illegal_customer_seq"][-1]["end"])
        max_t = change_to_std_time(max_t)
        machine_sequence = [0] * (max_t + 1)
        machine_pointer = 0
        customer_sequence = [0] * (max_t + 1)
        no_change_customer_timestamp_set = set()
        # 第二步, 整理每一个customer样本, 进行时序标注
        for inner_item in conv_item["customer_seq"]:
            s_t = change_to_std_time(inner_item["start"])
            e_t = change_to_std_time(inner_item["end"])
            no_change_customer_timestamp_set.add(s_t)
            no_change_customer_timestamp_set.add(e_t)
            for i in range(s_t, e_t + 1):
                customer_sequence[i] = 1
        # 第三步, 把中间断句等待的空白时间去除
        for illegal_item in conv_item["illegal_customer_seq"]:
            s_t = change_to_std_time(illegal_item["start"])
            e_t = change_to_std_time(illegal_item["end"])
            for i in range(s_t, e_t + 1):
                if i not in no_change_customer_timestamp_set:
                    customer_sequence[i] = 0
        # 第四步, 将需要预测的样本构建
        for inner_item in conv_item["customer_seq"]:
            while machine_pointer < len(conv_item["machine_seq"]) and conv_item["machine_seq"][machine_pointer]["start"] < inner_item["end"]:
                s_t = change_to_std_time(conv_item["machine_seq"][machine_pointer]["start"])
                e_t = change_to_std_time(conv_item["machine_seq"][machine_pointer]["end"])
                if e_t - s_t >= (max_machine_duration // segment_ms):
                    s_t = e_t - (max_machine_duration // segment_ms)
                for i in range(s_t, e_t + 1):
                    machine_sequence[i] = 1
                machine_pointer += 1
            e_t = change_to_std_time(inner_item["end"])
            if inner_item["need_predict"]:
                if e_t + 1 < window_size:
                    sample_machine_sequence = [0] * (window_size - e_t - 1) + machine_sequence[:e_t + 1]
                    sample_customer_sequence = [0] * (window_size - e_t - 1) + customer_sequence[:e_t + 1]
                else:
                    sample_machine_sequence = machine_sequence[e_t + 1 - window_size: e_t + 1]
                    sample_customer_sequence = customer_sequence[e_t + 1 - window_size: e_t + 1]
                append_item = {
                    "machine_sequence": sample_machine_sequence, "customer_sequence": sample_customer_sequence,
                    "label": inner_item["label"]
                }
                if inner_item["is_valid"]:
                    valid_data.append(append_item)
    return valid_data


# 建立前向传播计算图
def series_model(input_machine_ids, input_customer_ids):
    # 构建两个embedding矩阵
    with tf.variable_scope("embedding"):
        machine_embedding_matrix = tf.get_variable(
            "machine_embedding", initializer=np.array(np_weights_map["machine_embedding.weight"]),
            trainable=False, dtype=tf.float32
        )
        customer_embedding_matrix = tf.get_variable(
            "customer_embedding", initializer=np.array(np_weights_map["customer_embedding.weight"]),
            trainable=False, dtype=tf.float32
        )
        machine_embed = tf.nn.embedding_lookup(machine_embedding_matrix, input_machine_ids)
        customer_embed = tf.nn.embedding_lookup(customer_embedding_matrix, input_customer_ids)

    input_feature = tf.concat([machine_embed, customer_embed], axis=-1)

    # RNN层
    with tf.variable_scope("bi-gru"):
        forward_state, backward_state = tf.zeros(shape=(1, hidden_size), dtype=tf.float32), tf.zeros(shape=(1, hidden_size), dtype=tf.float32)
        for i in range(seq_len):
            with tf.variable_scope("forward"):
                with tf.variable_scope("gate_i", reuse=tf.AUTO_REUSE):
                    gate_i = fully_connected(
                        inputs=input_feature[:, i, :], num_outputs=3 * hidden_size, activation_fn=None, trainable=False,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map["rnn_encoder.weight_ih_l0"]), dtype=tf.float32),
                        biases_initializer=tf.constant_initializer(np.transpose(np_weights_map["rnn_encoder.bias_ih_l0"]), dtype=tf.float32),
                    )
                with tf.variable_scope("gate_h", reuse=tf.AUTO_REUSE):
                    gate_h = fully_connected(
                        inputs=forward_state, num_outputs=3 * hidden_size, activation_fn=None, trainable=False,
                        weights_initializer=tf.constant_initializer(np.transpose(np_weights_map["rnn_encoder.weight_hh_l0"]), dtype=tf.float32),
                        biases_initializer=tf.constant_initializer(np.transpose(np_weights_map["rnn_encoder.bias_hh_l0"]), dtype=tf.float32)
                    )
                i_r, i_i, i_n = tf.split(gate_i, num_or_size_splits=3, axis=-1)
                h_r, h_i, h_n = tf.split(gate_h, num_or_size_splits=3, axis=-1)
                reset_gate = tf.sigmoid(i_r + h_r)
                input_gate = tf.sigmoid(i_i + h_i)
                update_gate = tf.tanh(i_n + (reset_gate * h_n))
                forward_state = update_gate + input_gate * (forward_state - update_gate)

    output_feature = forward_state
    with tf.variable_scope("classifier"):
        logits = fully_connected(
            inputs=output_feature, num_outputs=2, activation_fn=None, trainable=False,
            weights_initializer=tf.constant_initializer(np.transpose(np_weights_map["classify_layer.weight"]), dtype=tf.float32),
            biases_initializer=tf.constant_initializer(np_weights_map["classify_layer.bias"], dtype=tf.float32)
        )
        prob = tf.nn.softmax(logits, axis=-1)
    return prob


embedding_size, hidden_size, seq_len = 64, 128, 150

input_machine_ids_holder = tf.placeholder(name="input_machine_ids", dtype=tf.int32, shape=[1, seq_len])
input_customer_ids_holder = tf.placeholder(name="input_customer_ids", dtype=tf.int32, shape=[1, seq_len])

prob = series_model(input_machine_ids_holder, input_customer_ids_holder)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

with open("data/time_series_data_no_interrupt.pkl", "rb") as f:
    data = pickle.load(f)
valid_data = preprocess_all_data(data, 30, 200, 3000)

print("验证数据量:%d" % len(valid_data))
print("验证开始时间: %s" % str(datetime.datetime.now()))

# 验证在负样本视角下, 不同预测阈值的结果, 0.4~0.6
y_true = []  # 将0切成1, 1切成0
y_pred_040, y_pred_042, y_pred_044, y_pred_046, y_pred_048 = [], [], [], [], []
y_pred_050, y_pred_052, y_pred_054, y_pred_056, y_pred_058, y_pred_060 = [], [], [], [], [], []
for item in valid_data:
    predict_prob = sess.run(prob, feed_dict={
        input_machine_ids_holder: [item["machine_sequence"]],
        input_customer_ids_holder: [item["customer_sequence"]]
    })
    y_true.append(1 - item["label"])
    if predict_prob[0][0] >= 0.40:
        y_pred_040.append(1)
    else:
        y_pred_040.append(0)
    if predict_prob[0][0] >= 0.42:
        y_pred_042.append(1)
    else:
        y_pred_042.append(0)
    if predict_prob[0][0] >= 0.44:
        y_pred_044.append(1)
    else:
        y_pred_044.append(0)
    if predict_prob[0][0] >= 0.46:
        y_pred_046.append(1)
    else:
        y_pred_046.append(0)
    if predict_prob[0][0] >= 0.48:
        y_pred_048.append(1)
    else:
        y_pred_048.append(0)
    if predict_prob[0][0] >= 0.50:
        y_pred_050.append(1)
    else:
        y_pred_050.append(0)
    if predict_prob[0][0] >= 0.52:
        y_pred_052.append(1)
    else:
        y_pred_052.append(0)
    if predict_prob[0][0] >= 0.54:
        y_pred_054.append(1)
    else:
        y_pred_054.append(0)
    if predict_prob[0][0] >= 0.56:
        y_pred_056.append(1)
    else:
        y_pred_056.append(0)
    if predict_prob[0][0] >= 0.58:
        y_pred_058.append(1)
    else:
        y_pred_058.append(0)
    if predict_prob[0][0] >= 0.60:
        y_pred_060.append(1)
    else:
        y_pred_060.append(0)
print("验证结束时间: %s" % str(datetime.datetime.now()))
print("\n")
print("在阈值为0.40的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_040), recall_score(y_true=y_true, y_pred=y_pred_040),
    f1_score(y_true=y_true, y_pred=y_pred_040)
))
print("在阈值为0.42的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_042), recall_score(y_true=y_true, y_pred=y_pred_042),
    f1_score(y_true=y_true, y_pred=y_pred_042)
))
print("在阈值为0.44的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_044), recall_score(y_true=y_true, y_pred=y_pred_044),
    f1_score(y_true=y_true, y_pred=y_pred_044)
))
print("在阈值为0.46的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_046), recall_score(y_true=y_true, y_pred=y_pred_046),
    f1_score(y_true=y_true, y_pred=y_pred_046)
))
print("在阈值为0.48的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_048), recall_score(y_true=y_true, y_pred=y_pred_048),
    f1_score(y_true=y_true, y_pred=y_pred_048)
))
print("在阈值为0.50的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_050), recall_score(y_true=y_true, y_pred=y_pred_050),
    f1_score(y_true=y_true, y_pred=y_pred_050)
))
print("在阈值为0.52的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_052), recall_score(y_true=y_true, y_pred=y_pred_052),
    f1_score(y_true=y_true, y_pred=y_pred_052)
))
print("在阈值为0.54的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_054), recall_score(y_true=y_true, y_pred=y_pred_054),
    f1_score(y_true=y_true, y_pred=y_pred_054)
))
print("在阈值为0.56的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_056), recall_score(y_true=y_true, y_pred=y_pred_056),
    f1_score(y_true=y_true, y_pred=y_pred_056)
))
print("在阈值为0.58的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_058), recall_score(y_true=y_true, y_pred=y_pred_058),
    f1_score(y_true=y_true, y_pred=y_pred_058)
))
print("在阈值为0.60的情况下, 负样本视角, 预测精确率为:%.3f, 召回率为:%.3f, F1-score为:%.3f" % (
    precision_score(y_true=y_true, y_pred=y_pred_060), recall_score(y_true=y_true, y_pred=y_pred_060),
    f1_score(y_true=y_true, y_pred=y_pred_060)
))

# 最后一步, 保存为saved_model格式, 修改signature为"test_signature"
inputs = {
    "input_machine_ids:0": tf.saved_model.utils.build_tensor_info(input_machine_ids_holder),
    "input_customer_ids:0": tf.saved_model.utils.build_tensor_info(input_customer_ids_holder)
}
outputs = {"classifier/Softmax:0": tf.saved_model.utils.build_tensor_info(prob)}

output_model = "./saved_model"
signature = (tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
))

b = tf.saved_model.builder.SavedModelBuilder(output_model)
b.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={"test_signature": signature})
b.save()
