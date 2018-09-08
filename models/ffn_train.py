# -*- coding: utf-8 -*-
"""
@created on: 8/28/18,
@author: Vivek A Gupta,

Description:

..todo::
"""
import sys
sys.path.append('/home/vivek.gupta/maxlife')

import tensorflow as tf
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
import datetime
from utils.data_utils import calculate_decile, calculate_gini_score_manual, calculate_precision_recall_curve, calculate_confusion_matrix
import numpy as np
import os

trans_train_path = sys.argv[1]
# trans_train_path = "/Users/vivek/sample.csv"
trans_test_path = sys.argv[2]
# trans_test_path = "/Users/vivek/sample.csv"

model_name = sys.argv[3]
# model_name = "ffn_model"

learning_rate = 0.001
epochs = 100
batch_size = 512
display_count = 1000
split_ratio = [100, 0, 0]

ignore_col_list = ["POL_ID", "DATA_MONTH", "CLI_INCOME_TODAY",
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
"TB_POL_CSTAT_CD_C",
"TPP_INSURED_MARITAL_STS_Others",
"TPP_INSURED_MARITAL_STS_Married",
"TPP_INSURED_MARITAL_STS_Single",
 "TPP_INSURED_GENDER_FEMALE",
                   "TPP_INSURED_GENDER_MALE",
                   "TPP_INSURED_GENDER_null",
"TPP_INSURED_INDUSTRY_missing",
"TPP_INSURED_INDUSTRY_high",
"TPP_INSURED_INDUSTRY_medium",
"TPP_INSURED_INDUSTRY_low",
"TPP_INSURED_INDUSTRY_others",
"TPP_INSURED_EDU_(A)Illiterate",
"TPP_INSURED_EDU_(D)Others",
"TPP_INSURED_EDU_(C)Grad & above",
"TPP_INSURED_EDU_(B)Schooling", "TPP_INSURED_INCOME"]

print("Reading the data...")
trans_train_data, trans_train_label, _, _, _, _ = read_csv(trans_train_path, split_ratio=split_ratio, header=True, ignore_cols=ignore_col_list, output_label="Lapse_Flag")
trans_test_data, trans_test_label, _, _, _, _ = read_csv(trans_test_path, split_ratio=split_ratio, header=True, ignore_cols=ignore_col_list, output_label="Lapse_Flag")

print(trans_train_data[0])
print(trans_train_label[0])

pos_weight = len(trans_train_label)/sum(trans_train_label)

print("Train Data Size - ", len(trans_train_data))
print("Test Data Size - ", len(trans_test_data))

print("Creating batches...")

# train_x = divide_batches_gen(trans_train_data, batch_size)
train_y = divide_batches(trans_train_label, batch_size)

# test_x = divide_batches_gen(trans_test_data, batch_size)
test_y = divide_batches(trans_test_label, batch_size)

train_batch_size = len(train_y)
test_batch_size = len(test_y)

# Placeholders.
with tf.name_scope("placeholders"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, len(trans_train_data[0])], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, len(trans_train_label[0])], name="output")
    z = tf.placeholder(dtype=tf.float32, shape=[], name="z")
    lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


# Model.
def model(x, lr):
    with tf.name_scope("ffn"):

        trans_weights = {
            'w_h1': tf.get_variable(name="w_h1", shape=[len(trans_train_data[0]), 100], initializer=tf.contrib.layers.xavier_initializer()),
            'w_h2': tf.get_variable(name="w_h2", shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer()),
            'w_h3': tf.get_variable(name="w_h3", shape=[50, 25], initializer=tf.contrib.layers.xavier_initializer()),
            'w_out': tf.get_variable(name="w_out", shape=[25, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        trans_bias = {
            'b_h1': tf.get_variable(name="b_h1", shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
            'b_h2': tf.get_variable(name="b_h2", shape=[50], initializer=tf.contrib.layers.xavier_initializer()),
            'b_h3': tf.get_variable(name="b_h3", shape=[25], initializer=tf.contrib.layers.xavier_initializer()),
            'b_out': tf.get_variable(name="b_out", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        }

        h1 = tf.add(tf.matmul(x, trans_weights['w_h1']), trans_bias['b_h1'])
        h1 = tf.nn.sigmoid(h1, name="trans_h1")

        h2 = tf.add(tf.matmul(h1, trans_weights['w_h2']), trans_bias['b_h2'])
        h2 = tf.nn.sigmoid(h2, name="trans_h2")

        h3 = tf.add(tf.matmul(h2, trans_weights['w_h3']), trans_bias['b_h3'])
        h3 = tf.nn.sigmoid(h3, name="trans_h3")

        trans_output = tf.add(tf.matmul(h3, trans_weights['w_out']), trans_bias['b_out'], name="trans_output")

    return trans_output


y_ = model(x, lr)
tf.add_to_collection("y_", y_)


# Loss.
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_, targets=y, pos_weight=pos_weight))
    tf.add_to_collection("loss", loss)

# Optimizer.
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    tf.add_to_collection("optimizer", optimizer)

# Create summary for loss.
train_loss_summ = tf.summary.scalar("train_batch_loss", loss)
test_loss_summ = tf.summary.scalar("test_batch_loss", loss)

train_avg_loss_summ = tf.summary.scalar("train_avg_loss", z)
test_avg_loss_summ = tf.summary.scalar("test_avg_loss", z)

# Create summary for gini.
train_gini_summ = tf.summary.scalar("train_gini", z)
test_gini_summ = tf.summary.scalar("test_gini", z)

# Create summary for decile.
train_decile_summ = tf.summary.scalar("train_decile", z)
test_decile_summ = tf.summary.scalar("test_decile", z)

logdir = "../tensorboard/ffn_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

saved_model_dir ="../maxlife_models/"
if not os.path.isdir(saved_model_dir):
    os.mkdir(saved_model_dir)


init = tf.global_variables_initializer()

# Model saver
model_saver = tf.train.Saver()

with tf.device("/GPU:0"):
    with tf.Session() as sess:
        sess.run(init)

        # model_saver.restore(sess, tf.train.latest_checkpoint('maxlife_models/'))

        writer = tf.summary.FileWriter(logdir, sess.graph)
        writer.add_graph(sess.graph)

        train_count = 0
        test_count = 0

        sess.run(lr, feed_dict={lr: learning_rate})

        for i in range(1, epochs + 1):

            train_x = divide_batches_gen(trans_train_data, batch_size)
            test_x = divide_batches_gen(trans_test_data, batch_size)

            # Train Data.
            count = 0
            train_loss = 0

            for train_data, train_label in zip(train_x, train_y):
                train_count += 1
                count += 1

                _, l = sess.run([optimizer, loss], feed_dict={x: train_data, y: train_label, lr: learning_rate})

                train_loss += l

                if count % display_count == 0:
                    train_summary = sess.run(train_loss_summ,
                                                     feed_dict={x: train_data, y: train_label})
                    writer.add_summary(train_summary, train_count)
                    print("Train Batch Count: ", count)
                    print("Train Iter Loss: ", l)

            # Test Data.
            count = 0
            test_loss = 0

            print("\n")

            for test_data, test_label in zip(test_x, test_y):
                count += 1
                test_count += 1

                l = sess.run(loss, feed_dict={x: test_data, y: test_label})

                test_loss += l

                if count % display_count == 0:
                    test_summary = sess.run(test_loss_summ, feed_dict={x: test_data, y: test_label})
                    writer.add_summary(test_summary, test_count)
                    print("Test Batch Count: ", count)
                    print("Test Iter Loss: ", l)

            train_x = divide_batches_gen(trans_train_data, batch_size)
            test_x = divide_batches_gen(trans_test_data, batch_size)

            # Calculate decile.
            train_predictions = []
            for train_data in train_x:
                model_prediction = sess.run(y_, feed_dict={x: train_data})
                train_predictions.append(temp for temp in model_prediction)

            test_predictions = []
            for test_data in test_x:
                model_prediction = sess.run(y_, feed_dict={x: test_data})
                test_predictions.append(temp for temp in model_prediction)

            train_predictions = [item for sublist in train_predictions for item in sublist]
            test_predictions = [item for sublist in test_predictions for item in sublist]

            train_decile_score = calculate_decile(train_predictions, list(trans_train_label))
            test_decile_score = calculate_decile(test_predictions, list(trans_test_label))

            trans_train_label = trans_train_label.reshape(trans_train_label.shape[0],)
            train_predictions = np.asarray(train_predictions)
            train_predictions = train_predictions.reshape(train_predictions.shape[0],)

            trans_test_label = trans_test_label.reshape(trans_test_label.shape[0],)
            test_predictions = np.asarray(test_predictions)
            test_predictions = test_predictions.reshape(test_predictions.shape[0],)

            train_gini = calculate_gini_score_manual(trans_train_label, train_predictions)
            test_gini = calculate_gini_score_manual(trans_test_label, test_predictions)

            precision, recall, threshold = calculate_precision_recall_curve(trans_test_label, test_predictions)
            #con_mat = calculate_confusion_matrix(trans_test_label, test_predictions, threshold)

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
            #print("\nConfustion Matrix")
            #print(con_mat)
            #print("\n\n")

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

            model_saver.save(sess, '../maxlife_models/' + model_name + '/ffn_model', global_step=i)
