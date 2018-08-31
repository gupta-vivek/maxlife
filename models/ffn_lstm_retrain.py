# -*- coding: utf-8 -*-
"""
@created on: 8/31/18,
@author: Vivek A Gupta,

Description:

..todo::
"""

import sys

sys.path.append('/home/vivek.gupta/maxlife')

import tensorflow as tf
from tensorflow.contrib import rnn
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
import datetime
from utils.data_utils import calculate_decile, calculate_gini_score_manual, calculate_precision_recall_curve, \
    calculate_confusion_matrix
import numpy as np
import os

model_name = sys.argv[1]

ffn_train_path = sys.argv[2]
ffn_test_path = sys.argv[3]

lstm_train_path = sys.argv[4]
lstm_test_path = sys.argv[5]

learning_rate = 0.001
keep_probability = 0.7
epochs = 200
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
                   "TPP_POL_PPT_2_0", "TPP_POL_PPT_3_0",
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
                   "LTST_SERV_CALL_TYP_null"]

print("Reading the data...")
ffn_train_data, ffn_train_label, _, _, _, _ = read_csv(ffn_train_path, split_ratio=split_ratio, header=True,
                                                       ignore_cols=ignore_col_list, output_label="Lapse_Flag")
lstm_train_data, _, _, _, _, _ = read_csv(lstm_train_path, split_ratio=split_ratio, header=True,
                                          ignore_cols=["POL_ID", "DATA_MONTH", "TB_POL_BILL_MODE_CD", "MI"],
                                          output_label="Lapse_Flag")
# lstm_train_data, _, _, _, _, _ = read_csv(lstm_train_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")

ffn_test_data, ffn_test_label, _, _, _, _ = read_csv(ffn_test_path, split_ratio=split_ratio, header=True,
                                                     ignore_cols=ignore_col_list, output_label="Lapse_Flag")
lstm_test_data, _, _, _, _, _ = read_csv(lstm_test_path, split_ratio=split_ratio, header=True,
                                         ignore_cols=["POL_ID", "DATA_MONTH", "TB_POL_BILL_MODE_CD", "MI"],
                                         output_label="Lapse_Flag")
# lstm_test_data, _, _, _, _, _ = read_csv(lstm_test_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")

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
print("Test Data Size - ", len(ffn_test_data))

print("Creating batches...")

# train_x = divide_batches_gen(ffn_train_data, batch_size)
train_y = divide_batches(ffn_train_label, batch_size)

# test_x = divide_batches_gen(ffn_test_data, batch_size)
test_y = divide_batches(ffn_test_label, batch_size)

train_batch_size = len(train_y)
test_batch_size = len(test_y)

logdir = "../tensorboard/ffn_lstm_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

saved_model_dir = "../maxlife_models/"
if not os.path.isdir(saved_model_dir):
    os.mkdir(saved_model_dir)

saved_model = saved_model_dir + model_name
ckpt = tf.train.latest_checkpoint(saved_model)
filename = ".".join([ckpt, 'meta'])
model_saver = tf.train.import_meta_graph(filename)
# init = tf.global_variables_initializer()

# Model saver
# model_saver = tf.train.Saver()

with tf.device("/GPU:0"):
    with tf.Session() as sess:
        # sess.run(init)

        # model_saver.restore(sess, tf.train.latest_checkpoint('maxlife_models/'))

        model_saver.restore(sess, ckpt)
        previous_count = int(filename.split('-')[1].split('.')[0])

        writer = tf.summary.FileWriter(logdir, sess.graph)
        writer.add_graph(sess.graph)

        graph = tf.get_default_graph()

        x_ffn = graph.get_tensor_by_name('placeholders/input_ffm:0')
        x_lstm = graph.get_tensor_by_name('placeholders/input_lstm:0')
        y = graph.get_tensor_by_name('placeholders/output:0')
        z = graph.get_tensor_by_name('placeholders/z:0')
        lr = graph.get_tensor_by_name('placeholders/lr:0')
        kp = graph.get_tensor_by_name('placeholders/kp:0')

        y_ = tf.get_collection("y_")[0]
        optimizer = tf.get_collection("optimizer")[0]
        loss = tf.get_collection("loss")[0]

        train_loss_summ = graph.get_tensor_by_name("train_batch_loss:0")
        test_loss_summ = graph.get_tensor_by_name("test_batch_loss:0")
        train_avg_loss_summ = graph.get_tensor_by_name("train_avg_loss:0")
        test_avg_loss_summ = graph.get_tensor_by_name("train_avg_loss:0")

        train_gini_summ = graph.get_tensor_by_name("train_gini:0")
        test_gini_summ = graph.get_tensor_by_name("test_gini:0")

        train_decile_summ = graph.get_tensor_by_name("train_decile:0")
        test_decile_summ = graph.get_tensor_by_name("test_decile:0")

        # writer = tf.summary.FileWriter(logdir, sess.graph)
        # writer.add_graph(sess.graph)

        train_count = 0
        test_count = 0

        sess.run(lr, feed_dict={lr: learning_rate})

        for i in range(1, epochs + 1):

            train_ffn_x = divide_batches_gen(ffn_train_data, batch_size)
            test_ffn_x = divide_batches_gen(ffn_test_data, batch_size)

            train_lstm_x = divide_batches(lstm_train_data, batch_size)
            test_lstm_x = divide_batches(lstm_test_data, batch_size)

            # Train Data.
            count = 0
            train_loss = 0

            for train_data_ffn, train_data_lstm, train_label in zip(train_ffn_x, train_lstm_x, train_y):
                train_count += 1
                count += 1

                _, l = sess.run([optimizer, loss],
                                feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, y: train_label,
                                           lr: learning_rate, kp: keep_probability})

                train_loss += l

                if count % display_count == 0:
                    train_summary = sess.run(train_loss_summ,
                                             feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, y: train_label,
                                                        kp: 1.0})
                    writer.add_summary(train_summary, train_count)
                    print("Train Batch Count: ", count)
                    print("Train Iter Loss: ", l)

            # Test Data.
            count = 0
            test_loss = 0

            print("\n")

            for test_data_ffn, test_data_lstm, test_label in zip(test_ffn_x, test_lstm_x, test_y):
                count += 1
                test_count += 1

                l = sess.run(loss, feed_dict={x_ffn: test_data_ffn, x_lstm: test_data_lstm, y: test_label, kp: 1.0})

                test_loss += l

                if count % display_count == 0:
                    test_summary = sess.run(test_loss_summ,
                                            feed_dict={x_ffn: test_data_ffn, x_lstm: test_data_lstm, y: test_label,
                                                       kp: 1.0})
                    writer.add_summary(test_summary, test_count)
                    print("Test Batch Count: ", count)
                    print("Test Iter Loss: ", l)

            train_ffn_x = divide_batches_gen(ffn_train_data, batch_size)
            test_ffn_x = divide_batches_gen(ffn_test_data, batch_size)

            train_lstm_x = divide_batches(lstm_train_data, batch_size)
            test_lstm_x = divide_batches(lstm_test_data, batch_size)

            # Calculate decile.
            train_predictions = []
            for train_data_ffn, train_data_lstm in zip(train_ffn_x, train_lstm_x):
                model_prediction = sess.run(y_, feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, kp: 1.0})
                train_predictions.append(temp for temp in model_prediction)

            test_predictions = []
            for test_data_ffn, test_data_lstm in zip(test_ffn_x, test_lstm_x):
                model_prediction = sess.run(y_, feed_dict={x_ffn: test_data_ffn, x_lstm: test_data_lstm, kp: 1.0})
                test_predictions.append(temp for temp in model_prediction)

            train_predictions = [item for sublist in train_predictions for item in sublist]
            test_predictions = [item for sublist in test_predictions for item in sublist]

            train_decile_score = calculate_decile(train_predictions, list(ffn_train_label))
            test_decile_score = calculate_decile(test_predictions, list(ffn_test_label))

            ffn_train_label = ffn_train_label.reshape(ffn_train_label.shape[0], )
            train_predictions = np.asarray(train_predictions)
            train_predictions = train_predictions.reshape(train_predictions.shape[0], )

            ffn_test_label = ffn_test_label.reshape(ffn_test_label.shape[0], )
            test_predictions = np.asarray(test_predictions)
            test_predictions = test_predictions.reshape(test_predictions.shape[0], )

            train_gini = calculate_gini_score_manual(ffn_train_label, train_predictions)
            test_gini = calculate_gini_score_manual(ffn_test_label, test_predictions)

            precision, recall, threshold = calculate_precision_recall_curve(ffn_test_label, test_predictions)
            # con_mat = calculate_confusion_matrix(ffn_test_label, test_predictions, threshold)

            print("\n\nEpoch:  ", i)
            print("Loss")
            print("Train: ", train_loss / train_batch_size)
            print("Test: ", test_loss / test_batch_size)
            print("Decile")
            print("Train: ", train_decile_score)
            print("Test: ", test_decile_score)
            print("Gini")
            print("Train: ", train_gini)
            print("Test: ", test_gini)
            print("\nPrecision: ", precision)
            print("Recall: ", recall)
            print("Threshold: ", threshold)
            print("\nConfustion Matrix")

            for t in threshold[:1]:
                print("thresh: ", t)
                con_mat = calculate_confusion_matrix(ffn_test_label, test_predictions, t)
                print(con_mat)
                print("\n")

            print('Default threshold - 0.5')
            con_mat = calculate_confusion_matrix(ffn_test_label, test_predictions, 0.5)
            print(con_mat)

            z_temp = sess.run(train_avg_loss_summ, feed_dict={z: train_loss / train_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_avg_loss_summ, feed_dict={z: test_loss / test_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(train_decile_summ, feed_dict={z: train_decile_score})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_decile_summ, feed_dict={z: test_decile_score})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(train_gini_summ, feed_dict={z: train_gini})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_gini_summ, feed_dict={z: test_gini})
            writer.add_summary(z_temp, i)

            model_saver.save(sess, '../maxlife_models/' + model_name + '/ffn_lstm_model', global_step=i)
