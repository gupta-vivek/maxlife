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
from tensorflow.contrib import rnn
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
import datetime
from utils.data_utils import calculate_decile, calculate_gini_score_manual, calculate_precision_recall_curve, calculate_confusion_matrix
import numpy as np
import os

model_name = sys.argv[1]

ffn_train_path = sys.argv[2]
ffn_test_path = sys.argv[3]

lstm_train_path = sys.argv[4]
lstm_test_path = sys.argv[5]

learning_rate = 0.001
keep_probability = 0.5
epochs = 200
batch_size = 512
display_count = 1000
split_ratio = [100, 0, 0]

print("Reading the data...")
ffn_train_data, ffn_train_label, _, _, _, _ = read_csv(ffn_train_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")
lstm_train_data, _, _, _, _, _ = read_csv(lstm_train_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH", "TB_POL_BILL_MODE_CD", "MI"], output_label="Lapse_Flag")
# lstm_train_data, _, _, _, _, _ = read_csv(lstm_train_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")

ffn_test_data, ffn_test_label, _, _, _, _ = read_csv(ffn_test_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")
lstm_test_data, _, _, _, _, _ = read_csv(lstm_test_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH", "TB_POL_BILL_MODE_CD", "MI"], output_label="Lapse_Flag")
# lstm_test_data, _, _, _, _, _ = read_csv(lstm_test_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH"], output_label="Lapse_Flag")

print("ffn data")
print(ffn_train_data[0])
print(len(ffn_train_data[0]))
print(ffn_train_label[0])
print(len(ffn_train_label[0]))

print("lstm data")
print(lstm_train_data[0])
print(len(lstm_train_data[0]))

pos_weight = len(ffn_train_label)/sum(ffn_train_label)

print("Train Data Size - ", len(ffn_train_data))
print("Test Data Size - ", len(ffn_test_data))

print("Creating batches...")

# train_x = divide_batches_gen(ffn_train_data, batch_size)
train_y = divide_batches(ffn_train_label, batch_size)

# test_x = divide_batches_gen(ffn_test_data, batch_size)
test_y = divide_batches(ffn_test_label, batch_size)

train_batch_size = len(train_y)
test_batch_size = len(test_y)

# Placeholders.
with tf.name_scope("placeholders"):
    x_ffn = tf.placeholder(dtype=tf.float32, shape=[None, len(ffn_train_data[0])], name="input_ffn")
    x_lstm = tf.placeholder(dtype=tf.float32, shape=[None, len(lstm_train_data[0])], name="input_lstm")
    y = tf.placeholder(dtype=tf.float32, shape=[None, len(ffn_train_label[0])], name="output")
    z = tf.placeholder(dtype=tf.float32, shape=[], name="z")
    lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    kp = tf.placeholder(dtype=tf.float32, shape=[], name="kp")


# Model.
def model(x_ffn, x_lstm, kp):

    with tf.name_scope("transaction"):
        reshape_trans_data = tf.reshape(x_lstm, name="reshape_trans", shape=[-1, len(lstm_train_data[0]), 1])
        unstack_trans_data = tf.unstack(reshape_trans_data, name="unstack_trans", axis=1)

        lstm_cell = rnn.BasicLSTMCell(name="trans_lstm", num_units=24, activation=tf.nn.relu)
        trans_lstm, states = rnn.static_rnn(lstm_cell, unstack_trans_data, dtype=tf.float32)

        trans_weights = {
            'w_h1': tf.get_variable(name="lw_h1", shape=[24, 10], initializer=tf.contrib.layers.xavier_initializer())
        }

        trans_bias = {
            'b_h1': tf.get_variable(name="lb_h1", shape=[10], initializer=tf.contrib.layers.xavier_initializer())

        }

        h1 = tf.add(tf.matmul(trans_lstm[-1], trans_weights['w_h1']), trans_bias['b_h1'])
        h1_sig = tf.nn.sigmoid(h1, name="trans_h1")
        # h1 = tf.nn.dropout(h1, keep_prob=kp)

        # h2 = tf.add(tf.matmul(h1, trans_weights['w_h2']), trans_bias['b_h2'])
        # h2 = tf.nn.sigmoid(h2, name="trans_h2")

        # trans_output = tf.add(tf.matmul(h1, trans_weights['w_out']), trans_bias['b_out'], name="trans_output")
    with tf.name_scope("ffn"):

        ffn_weights = {
            'w_h1': tf.get_variable(name="fw_h1", shape=[len(ffn_train_data[0]),75], initializer=tf.contrib.layers.xavier_initializer()),
            'w_h2': tf.get_variable(name="fw_h2", shape=[75,25], initializer=tf.contrib.layers.xavier_initializer()),
            'w_h3': tf.get_variable(name="fw_h3", shape=[35, 10], initializer=tf.contrib.layers.xavier_initializer()),
            'w_out': tf.get_variable(name="fw_out", shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        ffn_bias = {
            'b_h1': tf.get_variable(name="fb_h1", shape=[75], initializer=tf.contrib.layers.xavier_initializer()),
            'b_h2': tf.get_variable(name="fb_h2", shape=[25], initializer=tf.contrib.layers.xavier_initializer()),
            'b_h3': tf.get_variable(name="fb_h3", shape=[10], initializer=tf.contrib.layers.xavier_initializer()),
            'b_out': tf.get_variable(name="fb_out", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        }



        f_h1 = tf.add(tf.matmul(x_ffn, ffn_weights['w_h1']), ffn_bias['b_h1'])
        f_h1 = tf.nn.sigmoid(f_h1, name="trans_h1")

        f_h2 = tf.add(tf.matmul(f_h1, ffn_weights['w_h2']), ffn_bias['b_h2'])
        f_h2 = tf.nn.sigmoid(f_h2, name="trans_h2")

        lstm_ffn_concat = tf.concat([h1_sig, f_h2], axis=1, name="concat_input")

        lstm_ffn_concat_output = tf.nn.sigmoid(tf.add(tf.matmul(lstm_ffn_concat, ffn_weights['w_h3']), ffn_bias['b_h3'], name="lstm_ffn_concat_output"))

        f_h3 = tf.add(tf.matmul(lstm_ffn_concat_output, ffn_weights['w_out']), ffn_bias['b_out'])
        # f_h3 = tf.nn.sigmoid(f_h3, name="trans_h3")

    # with tf.name_scope("final_output"):
    #     final_weights = {
    #         'w_out': tf.get_variable(name="w_out", shape=[2, 1], initializer=tf.contrib.layers.xavier_initializer())
    #     }
    #
    #     final_bias = {
    #         'b_out': tf.get_variable(name="b_out", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    #     }
    #     combined_output = tf.concat([ffn_output, trans_output], axis=1, name="combined_output")
    #     combined_output = tf.nn.sigmoid(combined_output)
    #     final_output = tf.add(tf.matmul(combined_output, final_weights['w_out']), final_bias['b_out'])

    return f_h3


y_ = model(x_ffn, x_lstm, kp)
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

logdir = "../tensorboard/ffn_lstm_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

                _, l = sess.run([optimizer, loss], feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, y: train_label, lr: learning_rate, kp: keep_probability})

                train_loss += l

                if count % display_count == 0:
                    train_summary = sess.run(train_loss_summ,
                                                     feed_dict={x_ffn: train_data_ffn, x_lstm: train_data_lstm, y: train_label, kp: 1.0})
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
                    test_summary = sess.run(test_loss_summ, feed_dict={x_ffn: test_data_ffn, x_lstm: test_data_lstm, y: test_label, kp: 1.0})
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

            ffn_train_label = ffn_train_label.reshape(ffn_train_label.shape[0],)
            train_predictions = np.asarray(train_predictions)
            train_predictions = train_predictions.reshape(train_predictions.shape[0],)

            ffn_test_label = ffn_test_label.reshape(ffn_test_label.shape[0],)
            test_predictions = np.asarray(test_predictions)
            test_predictions = test_predictions.reshape(test_predictions.shape[0],)

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