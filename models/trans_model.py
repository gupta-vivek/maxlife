# -*- coding: utf-8 -*-
"""
@created on: 8/20/18,
@author: Vivek A Gupta,

Description:

..todo::
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
import datetime
from utils.data_utils import calculate_decile
import numpy as np


trans_train_path = "../data/trans_train.csv"
trans_test_path = "../data/trans_test.csv"

learning_rate = 0.001
epochs = 50
batch_size = 1000
display_count = 1
split_ratio = [100, 0, 0]

print("Reading the data...")
trans_train_data, trans_train_label, _, _, _, _ = read_csv(trans_train_path, split_ratio=split_ratio)
trans_test_data, trans_test_label, _, _, _, _ = read_csv(trans_test_path, split_ratio=split_ratio)

print("Train Data Size - ", len(trans_train_data))
print("Test Data Size - ", len(trans_test_data))

print("Splitting the data...")

# train_x = divide_batches_gen(trans_train_data, batch_size)
train_y = divide_batches(trans_train_label, batch_size)

# test_x = divide_batches_gen(trans_test_data, batch_size)
test_y = divide_batches(trans_test_label, batch_size)

train_batch_size = len(train_y)
test_batch_size = len(test_y)

# Placeholders.
with tf.name_scope("placeholders"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 24], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="output")


# Model.
def model(x):
    with tf.name_scope("transaction"):
        reshape_trans_data = tf.reshape(x, name="reshape_trans", shape=[-1, 24, 1])
        unstack_trans_data = tf.unstack(reshape_trans_data, name="unstack_trans", axis=1)

        lstm_cell = rnn.BasicLSTMCell(name="trans_lstm", num_units=1)
        trans_lstm, states = rnn.static_rnn(lstm_cell, unstack_trans_data, dtype=tf.float32)
        trans_lstm = tf.nn.tanh(trans_lstm)

        trans_weights = {
            'out': tf.random_normal([1, 1])
        }

        trans_bias = {
            'out': tf.random_normal([1])
        }

        trans_output = tf.add(tf.matmul(trans_lstm[-1], trans_weights['out']), trans_bias['out'], name="trans_output")

    return trans_output


y_ = model(x)

y_sig = tf.nn.sigmoid(y_)

# Loss.
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y))

# Optimizer.
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Accuracy.
with tf.name_scope("accuracy"):
    predicted_class = tf.greater(y_sig,0.5)
    correct = tf.equal(predicted_class, tf.equal(y,1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float')) * 100

# Create summary for loss.
train_loss_summ = tf.summary.scalar("train_batch_loss", loss)
test_loss_summ = tf.summary.scalar("test_batch_loss", loss)

z = tf.placeholder(tf.float64, shape=[])
train_avg_loss_summ = tf.summary.scalar("train_avg_loss", z)
test_avg_loss_summ = tf.summary.scalar("test_avg_loss", z)

# Create summary for accuracy.
train_accuracy_summ = tf.summary.scalar("train_batch_accuracy", accuracy)
test_accuracy_summ = tf.summary.scalar("test_batch_accuracy", accuracy)

train_avg_accuracy_summ = tf.summary.scalar("train_avg_accuracy", z)
test_avg_accuracy_summ = tf.summary.scalar("test_avg_accuracy", z)

# Create summary for decile.
train_decile_summ = tf.summary.scalar("train_decile", z)
test_decile_summ = tf.summary.scalar("test_decile", z)

summary_merged = tf.summary.merge_all()

logdir = "../tensorboard_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

init = tf.global_variables_initializer()

# Model saver
model_saver = tf.train.Saver()

with tf.device("/GPU:1"):
    with tf.Session() as sess:
        sess.run(init)

        # model_saver.restore(sess, tf.train.latest_checkpoint('maxlife_models/'))

        writer = tf.summary.FileWriter(logdir, sess.graph)
        writer.add_graph(sess.graph)

        train_count = 0
        test_count = 0

        for i in range(epochs):

            train_x = divide_batches_gen(trans_train_data, batch_size)
            test_x = divide_batches_gen(trans_test_data, batch_size)

            # Train Data.
            count = 0
            train_loss = 0
            train_acc = 0

            for train_data, train_label in zip(train_x, train_y):
                train_count += 1
                count += 1

                _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={x: train_data, y: train_label})

                train_loss += l
                train_acc += acc

                if count % display_count == 0:
                    train_summary = sess.run(train_loss_summ,
                                                     feed_dict={x: train_data, y: train_label})
                    writer.add_summary(train_summary, train_count)
                    print("Train Batch Count: ", count)
                    print("Train Iter Loss: ", l)
                    print("Train Iter Accuracy: ", acc)

            # Test Data.
            count = 0
            test_loss = 0
            test_acc = 0

            print("\n")

            for test_data, test_label in zip(test_x, test_y):
                count += 1
                test_count += 1

                l, acc = sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label})

                test_loss += l
                test_acc += acc

                if count % display_count == 0:
                    test_summary = sess.run(test_loss_summ, feed_dict={x: test_data, y: test_label})
                    writer.add_summary(test_summary, test_count)
                    print("Test Batch Count: ", count)
                    print("Test Iter Loss: ", l)
                    print("Test Iter Accuracy: ", acc)

            # Calculate decile.
            train_predictions = []
            for train_data in trans_train_data:
                model_prediction = sess.run(y_, feed_dict={x: [train_data]})
                train_predictions.append(model_prediction[0])

            test_predictions = []
            for test_data in trans_test_data:
                model_prediction = sess.run(y_, feed_dict={x: [test_data]})
                test_predictions.append(model_prediction[0])

            train_decile_score = calculate_decile(train_predictions, list(trans_train_label))
            test_decile_score = calculate_decile(test_predictions, list(trans_test_label))

            print("\n\nEpoch:  ", i)
            print("Loss")
            print("Train: ", train_loss / train_batch_size)
            print("Test: ", test_loss / test_batch_size)
            print("Accuracy")
            print("Train: ", train_acc / train_batch_size)
            print("Test: ", test_acc / test_batch_size)
            print("Decile")
            print("Train: ", train_decile_score)
            print("Test: ", test_decile_score)
            print("\n\n")

            z_temp = sess.run(train_avg_loss_summ, feed_dict={z: train_loss / train_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_avg_loss_summ, feed_dict={z: test_loss / test_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(train_avg_accuracy_summ, feed_dict={z: train_acc / train_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_avg_accuracy_summ, feed_dict={z: test_acc / test_batch_size})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(train_decile_summ, feed_dict={z: train_decile_score})
            writer.add_summary(z_temp, i)

            z_temp = sess.run(test_decile_summ, feed_dict={z: test_decile_score})
            writer.add_summary(z_temp, i)

            model_saver.save(sess, '../maxlife_models/trans_model', global_step=i)
