# -*- coding: utf-8 -*-
"""
@created on: 8/20/18,
@author: Vivek A Gupta,

Description:

..todo::
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from utils.file_utils import read_csv, divide_batches
import sys


trans_path = sys.argv[0]

learning_rate = 0.001
epochs = 100
batch_size = 10

USE_TRANS = True
ONE_DATA = False

data_placeholder_train = {}
data_placeholder_valid = {}
data_placeholder_test = {}

feed_dict_train = {}
final_train = []

concat_attr = []

if USE_TRANS:

    ONE_DATA = True

    trans_train_data, trans_train_label, trans_valid_data, trans_valid_label, trans_test_data, trans_test_label = read_csv(trans_path)

    data_placeholder_train['trans_train'] = trans_train_data
    # data_placeholder_valid['trans_valid'] = trans_valid_data
    # data_placeholder_test['trans_test'] = trans_test_data

    with tf.name_scope("transaction"):
        trans_input = tf.placeholder(shape=[None, 24], dtype=tf.float32, name="trans_input")
        reshape_trans_data = tf.reshape(trans_input, name="reshape_trans", shape=[-1, 24, 1])
        unstack_trans_data = tf.unstack(reshape_trans_data, name="unstack_trans", axis=1)

        lstm_cell = rnn.BasicLSTMCell(name="trans_lstm", num_units=24)
        trans_lstm, states = rnn.static_rnn(lstm_cell, unstack_trans_data, dtype=tf.float32)
        trans_lstm = tf.nn.tanh(trans_lstm)

        trans_weights = {
            'out': tf.random_normal([24, 10])
        }

        trans_bias = {
            'out': tf.random_normal([10])
        }

        trans_output = tf.add(tf.matmul(trans_lstm[-1], trans_weights['out']), trans_bias['out'], name="trans_output")

        concat_attr.append(trans_output)


if not ONE_DATA:
    raise Exception('All Attributes Are Off!')


with tf.name_scope("final_ffn"):
    concat_output = tf.concat(concat_attr, axis=1, name="concat_output")

    final_weights = {
        'w1': tf.random_normal([10, 10]),
        'out': tf.random_normal([10, 1])
    }

    final_bias = {
        'b1': tf.random_normal([10]),
        'out': tf.random_normal([1])
    }

    final_hidden = tf.add(tf.matmul(concat_output, final_weights['w1']), final_bias['b1'])
    final_hidden = tf.nn.sigmoid(final_hidden, name="final_hidden")
    final_output = tf.add(tf.matmul(final_hidden, final_weights['out']), final_bias['out'], name="final_output")


loss = tf.nn.sigmoid_cross_entropy_with_logits()
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

logdir = "tensorboard_dir/"

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Summary Writer.
    writer = tf.summary.FileWriter(logdir, sess.graph)
    writer.add_graph(sess.graph)

    sess.run()