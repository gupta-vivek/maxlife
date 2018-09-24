# -*- coding: utf-8 -*-
"""
@created on: 8/27/18,
@author: Vivek A Gupta,

Description:

..todo::
"""
import sys

sys.path.append('/home/vivek.gupta/maxlife')

import tensorflow as tf
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
from utils.data_utils import calculate_decile, calculate_gini_score_manual, calculate_precision_recall_curve, \
    calculate_confusion_matrix, sigmoid, Find_Optimal_Cutoff
import numpy as np
import os
import pandas as pd
import math

model_name = sys.argv[1]
ffn_train_path = sys.argv[2]
lstm_train_path = sys.argv[3]
output_file = sys.argv[4]

learning_rate = 0.001
keep_probability = 0.7
batch_size = 512
display_count = 1000
split_ratio = [100, 0, 0]

ignore_col_list_ffn = ["POL_ID", "DATA_MONTH", "RATIO_CLI__ACTIVE_ISSUED", "PLAN_AFYP_DERIV", "TB_POL_FUND_BALANCE"]
label_list = ["LAPSE_FLAG", "SURR_FLAG", "ETI_FLAG", "NO_FLAG"]
ignore_col_list_lstm = ["POL_ID", "DATA_MONTH"]

print("Reading the data...")
ffn_train_data, ffn_train_label, _, _, _, _ = read_csv(ffn_train_path, split_ratio=split_ratio, header=True,
                                                       ignore_cols=ignore_col_list_ffn, output_label=label_list, label_vector=True)
lstm_train_data, _, _, _, _, _ = read_csv(lstm_train_path, split_ratio=split_ratio, header=True,
                                          ignore_cols=ignore_col_list_lstm,
                                          output_label="Lapse_Flag")

print("ffn data")
print(ffn_train_data[0])
print(len(ffn_train_data[0]))
print(ffn_train_label[0])
print(len(ffn_train_label[0]))

print("lstm data")
print(lstm_train_data[0])
print(len(lstm_train_data[0]))

# pos_weight = len(ffn_train_label) / sum(ffn_train_label)

pos_weight = np.count_nonzero(ffn_train_label == 0) / np.count_nonzero(ffn_train_label == 1)

print("Train Data Size - ", len(ffn_train_data))

print("Creating batches...")

# train_x = divide_batches_gen(ffn_train_data, batch_size)
train_y = divide_batches(ffn_train_label, batch_size)

train_batch_size = len(train_y)


saved_model_dir ="../maxlife_models/"
if not os.path.isdir(saved_model_dir):
    os.mkdir(saved_model_dir)

saved_model = saved_model_dir + model_name
ckpt = tf.train.latest_checkpoint(saved_model)
filename = ".".join([ckpt, 'meta'])
model_saver = tf.train.import_meta_graph(filename, clear_devices=True)

with tf.device("/GPU:0"):
    with tf.Session() as sess:
        # sess.run(init)

        # model_saver.restore(sess, tf.train.latest_checkpoint('maxlife_models/'))

        model_saver.restore(sess, ckpt)
        #previous_count = int(filename.split('-')[1].split('.')[0])

        graph = tf.get_default_graph()

        x_ffn = graph.get_tensor_by_name('placeholders/input_ffn:0')
        x_lstm = graph.get_tensor_by_name('placeholders/input_lstm:0')
        y = graph.get_tensor_by_name('placeholders/output:0')
        z = graph.get_tensor_by_name('placeholders/z:0')
        lr = graph.get_tensor_by_name('placeholders/lr:0')
        kp = graph.get_tensor_by_name('placeholders/kp:0')
        y_ = tf.get_collection("y_")[0]


        # writer = tf.summary.FileWriter(logdir, sess.graph)
        # writer.add_graph(sess.graph)

        train_count = 0
        test_count = 0

        sess.run(lr, feed_dict={lr: learning_rate})

        train_ffn_x = divide_batches_gen(ffn_train_data, batch_size)
        train_lstm_x = divide_batches(lstm_train_data, batch_size)

        # Calculate decile.
        train_predictions = []
        for train_data_ffn, train_data_lstm in zip(train_ffn_x, train_lstm_x):
            model_prediction = sess.run(y_, feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, kp: 1.0})
            train_predictions.append(temp for temp in model_prediction)

        train_predictions = [item for sublist in train_predictions for item in sublist]

        train_decile_score = calculate_decile(np.asarray(train_predictions), ffn_train_label, decile_distribution=True)

        ffn_train_label = ffn_train_label.reshape(ffn_train_label.shape[0], )
        train_predictions = np.asarray(train_predictions)
        train_predictions = train_predictions.reshape(train_predictions.shape[0], )

        # train_gini = calculate_gini_score_manual(ffn_train_label, train_predictions)
        # precision, recall, threshold = calculate_precision_recall_curve(ffn_train_label, train_predictions)
        #con_mat = calculate_confusion_matrix(ffn_train_label, train_predictions, threshold)

        print("Decile: ", train_decile_score)
        # print("Gini: ", train_gini)
        # print("\nPrecision: ", precision)
        # print("Recall: ", recall)
        # print("Threshold: ", threshold)

        temp_df = pd.read_csv(lstm_train_path)
        df = pd.DataFrame()
        df['POL_ID'] = temp_df['POL_ID']
        df['DATA_MONTH'] = temp_df['DATA_MONTH']

        # y_pred = sigmoid(train_predictions)
        # y_pred[np.where(y_pred >= 0.5)] = 1.0
        # y_pred[np.where(y_pred < 0.5)] = 0.0
        # df['Predictions'] = y_pred
        df['Raw_Output'] = sigmoid(train_predictions)
        thrshold = Find_Optimal_Cutoff(ffn_train_label, df['Raw_Output'].values)[0]
        print("Optimal Threshold - ", thrshold)
        # con_mat = calculate_confusion_matrix(ffn_train_label, train_predictions, thrshold)
        # print("Confusion Matrix")
        # print(con_mat)
        df['Predicted_Flag'] = df.apply(lambda row: 1 if row['Raw_Output'] >= thrshold else 0, axis=1)
        # df['Label'] = ffn_train_label

        df.sort_values(by=['Raw_Output'], ascending=[False], inplace=True)
        len_df = len(df)
        len_10 = math.ceil(len_df / 10)
        len_10_x = len_10

        x = 0
        temp = [10 for i in range(len_10_x)]
        decile = np.asarray(temp)
        for i in reversed(range(9)):
            temp = [i + 1 for j in range(len_10_x)]
            decile = np.concatenate([decile, np.asarray(temp)], axis=0)
        df['Decile'] = decile[:len_df]
        df.to_csv(output_file, index=False)
