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

ignore_col_list = ["POL_ID", "DATA_MONTH", "TPP_POL_MED_NONMED_MED",
                   "TPP_POL_MED_NONMED_NON-MED",
                   "TPP_POL_MED_NONMED_null",
                   "TPP_POL_MED_NONMED_CLEAR", "TB_POL_RIDER_COUNT_0",
                   "TB_POL_RIDER_COUNT_1", "TB_POL_RIDER_COUNT_2",
                   "TB_POL_RIDER_COUNT_3", "TB_POL_RIDER_COUNT_4",
                   "TB_POL_RIDER_COUNT_5", "TB_POL_RIDER_COUNT_null",
                   "TB_POL_RIDER_COUNT_6", "TB_POL_RIDER_COUNT_7",
                   "TPP_POL_PPT_0_0", "TPP_POL_PPT_1_0",
                   "TPP_POL_PPT_2_0", "TPP_POL_PPT_3_0", "TPP_POL_PPT_3_NULL"
                   "AGENT_EDUCATION_null", "AGENT_EDUCATION_Post Graduation",
                   "AGENT_EDUCATION_Under Graduation",
                   "AGENT_EDUCATION_Doctorate",
                   "AGENT_EDUCATION_Intermediate",
                   "AGENT_EDUCATION_Graduation",
                   "AGT_VINTAGE_0_0",
                   "AGT_VINTAGE_1_0",
                   "AGT_VINTAGE_2_0",
                   "AGT_VINTAGE_3_0",
                   "AGT_VINTAGE_4_0",
                   "AGT_VINTAGE_5_0",
                   "AGT_VINTAGE_null",
                   "AGT_VINTAGE_6_0",
                   "GO_ZONE_West 2",
                   "GO_ZONE_null",
                   "GO_ZONE_East",
                   "GO_ZONE_North",
                   "GO_ZONE_West-Zone",
                   "GO_ZONE_South Zone",
                   "GO_ZONE_South_East and Central-Zone",
                   "GO_ZONE_South East West",
                   "GO_ZONE_West",
                   "GO_ZONE_East Zone",
                   "GO_ZONE_Others",
                   "GO_ZONE_West 1",
                   "GO_ZONE_South",
                   "GO_ZONE_North Zone",
                   "GO_ZONE_West Zone",
                   "GO_ZONE_North 2",
                   "GO_ZONE_North 1",
                   "NOMINEE_GENDER_FEMALE",
                   "NOMINEE_GENDER_MALE",
                   "NOMINEE_GENDER_null",
                   "TB_CLI_HNI_IND_L",
                   "TB_CLI_HNI_IND_Y",
                   "TB_CLI_HNI_IND_N",
                   "TB_CLI_HNI_IND_null",
                   "TPP_INSURED_SMOKER_FLG_N",
                   "TPP_INSURED_SMOKER_FLG_S",
                   "TPP_INSURED_SMOKER_FLG_null",
                   "TPP_INSURED_GENDER_FEMALE",
                   "TPP_INSURED_GENDER_MALE",
                   "TPP_INSURED_GENDER_null",
                   "INSURED_AGE_AT_ISSUE_0_0",
                   "INSURED_AGE_AT_ISSUE_1_0",
                   "INSURED_AGE_AT_ISSUE_2_0",
                   "INSURED_AGE_AT_ISSUE_3_0",
                   "INSURED_AGE_AT_ISSUE_4_0",
                   "INSURED_AGE_AT_ISSUE_5_0",
                   "INSURED_AGE_AT_ISSUE_null",
                   "INSURED_AGE_AT_ISSUE_6_0",
                   "LTST_SERV_CALL_TYP_POSITIVE",
                   "LTST_SERV_CALL_TYP_NEUTRAL",
                   "LTST_SERV_CALL_TYP_NEGATIVE",
                   "LTST_SERV_CALL_TYP_null",
                   "TB_POL_CSTAT_CD_PCRU",
                   "TB_POL_CSTAT_CD_5",
                   "TB_POL_CSTAT_CD_4",
                   "TB_POL_CSTAT_CD_1A",
                   "TB_POL_CSTAT_CD_PECC",
                   "TB_POL_CSTAT_CD_PCCU",
                   "TB_POL_CSTAT_CD_PCRC",
                   "TB_POL_CSTAT_CD_2",
                   "TB_POL_CSTAT_CD_M",
                   "TB_POL_CSTAT_CD_B",
                   "TB_POL_CSTAT_CD_D",
                   "TB_POL_CSTAT_CD_PERU",
                   "TB_POL_CSTAT_CD_Y",
                   "TB_POL_CSTAT_CD_1",
                   "TB_POL_CSTAT_CD_R",
                   "TB_POL_CSTAT_CD_A",
                   "TB_POL_CSTAT_CD_3",
                   "TB_POL_CSTAT_CD_E",
                   "TB_POL_CSTAT_CD_PERC",
                   "TB_POL_CSTAT_CD_C"]

ignore_col_list_ffn = ["POL_ID", "DATA_MONTH", "DUE_DATE"]
                       # "CLI_INCOME_TODAY"]
                   # "TB_POL_CSTAT_CD_PCRU",
                   # "TB_POL_CSTAT_CD_5",
                   # "TB_POL_CSTAT_CD_4",
                   # "TB_POL_CSTAT_CD_1A",
                   # "TB_POL_CSTAT_CD_PECC",
                   # "TB_POL_CSTAT_CD_PCCU",
                   # "TB_POL_CSTAT_CD_PCRC",
                   # "TB_POL_CSTAT_CD_2",
                   # "TB_POL_CSTAT_CD_M",
                   # "TB_POL_CSTAT_CD_B",
                   # "TB_POL_CSTAT_CD_D",
                   # "TB_POL_CSTAT_CD_PERU",
                   # "TB_POL_CSTAT_CD_Y",
                   # "TB_POL_CSTAT_CD_1",
                   # "TB_POL_CSTAT_CD_R",
                   # "TB_POL_CSTAT_CD_A",
                   # "TB_POL_CSTAT_CD_3",
                   # "TB_POL_CSTAT_CD_E",
                   # "TB_POL_CSTAT_CD_PERC",
                   # "TB_POL_CSTAT_CD_C",
                   # "TPP_INSURED_MARITAL_STS_Others",
                   # "TPP_INSURED_MARITAL_STS_Married",
                   # "TPP_INSURED_MARITAL_STS_Single",
                   # "TPP_INSURED_GENDER_FEMALE",
                   # "TPP_INSURED_GENDER_MALE",
                   # "TPP_INSURED_GENDER_null",
                   # "TPP_INSURED_INDUSTRY_missing",
                   # "TPP_INSURED_INDUSTRY_high",
                   # "TPP_INSURED_INDUSTRY_medium",
                   # "TPP_INSURED_INDUSTRY_low",
                   # "TPP_INSURED_INDUSTRY_others",
                   # "TPP_INSURED_EDU_(A)Illiterate",
                   # "TPP_INSURED_EDU_(D)Others",
                   # "TPP_INSURED_EDU_(C)Grad & above",
                   # "TPP_INSURED_EDU_(B)Schooling", "TPP_INSURED_INCOME"]

ignore_col_list_lstm = ["POL_ID", "DATA_MONTH"]

print("Reading the data...")
ffn_train_data, ffn_train_label, _, _, _, _ = read_csv(ffn_train_path, split_ratio=split_ratio, header=True,
                                                       ignore_cols=ignore_col_list_ffn, output_label="Lapse_Flag")
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

        train_decile_score = calculate_decile(train_predictions, list(ffn_train_label), decile_distribution=True)

        ffn_train_label = ffn_train_label.reshape(ffn_train_label.shape[0], )
        train_predictions = np.asarray(train_predictions)
        train_predictions = train_predictions.reshape(train_predictions.shape[0], )

        train_gini = calculate_gini_score_manual(ffn_train_label, train_predictions)
        # precision, recall, threshold = calculate_precision_recall_curve(ffn_train_label, train_predictions)
        #con_mat = calculate_confusion_matrix(ffn_train_label, train_predictions, threshold)

        print("Decile: ", train_decile_score)
        print("Gini: ", train_gini)
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
        con_mat = calculate_confusion_matrix(ffn_train_label, train_predictions, thrshold)
        print("Confusion Matrix")
        print(con_mat)
        df['Predicted_Flag'] = df.apply(lambda row: 1 if row['Raw_Output'] >= thrshold else 0, axis=1)
        # df['Label'] = ffn_train_label

        df.sort_values(by=['Raw_Output'], ascending=[False], inplace=True)
        len_df = len(df)
        len_10 = math.floor(len_df / 10)
        len_10_x = len_10

        x = 0
        temp = [10 for i in range(len_10_x)]
        decile = np.asarray(temp)
        for i in reversed(range(9)):
            temp = [i + 1 for j in range(len_10_x)]
            decile = np.concatenate([decile, np.asarray(temp)], axis=0)
        df['Decile'] = decile[:len_df]
        df.to_csv(output_file, index=False)
