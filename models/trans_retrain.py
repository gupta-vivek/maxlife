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
import datetime
from utils.file_utils import read_csv, divide_batches, divide_batches_gen
from utils.data_utils import calculate_decile, calculate_gini_score_manual
import numpy as np
import os

# trans_train_path = "../data/trans_new_train.csv"
trans_train_path = sys.argv[1]
# trans_test_path = "../data/trans_new_test.csv"
trans_test_path = sys.argv[2]

model_name = sys.argv[3]

learning_rate = 0.001
epochs = 10
batch_size = 512
display_count = 1000
split_ratio = [100, 0, 0]

print("Reading the data...")
trans_train_data, trans_train_label, _, _, _, _ = read_csv(trans_train_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH", "MODE_OF_PAYMENT", "MI"], output_label="Lapse_Flag")
trans_test_data, trans_test_label, _, _, _, _ = read_csv(trans_test_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH", "MODE_OF_PAYMENT", "MI"], output_label="Lapse_Flag")

print(trans_train_data[0])

print("Train Data Size - ", len(trans_train_data))
print("Test Data Size - ", len(trans_test_data))

print("Splitting the data...")

# train_x = divide_batches_gen(trans_train_data, batch_size)
train_y = divide_batches(trans_train_label, batch_size)

# test_x = divide_batches_gen(trans_test_data, batch_size)
test_y = divide_batches(trans_test_label, batch_size)

train_batch_size = len(train_y)
test_batch_size = len(test_y)


logdir = "../tensorboard/transaction_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

saved_model_dir ="../maxlife_models/"
if not os.path.isdir(saved_model_dir):
    os.mkdir(saved_model_dir)


with tf.Session() as sess:
    saved_model = saved_model_dir + model_name
    ckpt = tf.train.latest_checkpoint(saved_model)
    filename = ".".join([ckpt, 'meta'])
    previous_count = int(filename.split('-')[1].split('.')[0])
    model_saver = tf.train.import_meta_graph(filename)
    model_saver.restore(sess, ckpt)

    writer = tf.summary.FileWriter(logdir, sess.graph)
    writer.add_graph(sess.graph)

    graph = tf.get_default_graph()

    # for op in tf.get_default_graph().get_operations():
    #     print(str(op.name))

    x = graph.get_tensor_by_name('placeholders/input:0')
    y = graph.get_tensor_by_name('placeholders/output:0')
    z = graph.get_tensor_by_name('placeholders/z:0')

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

    train_count = 0
    test_count = 0

    for i in range(previous_count + 1, previous_count + epochs + 1):

        train_x = divide_batches_gen(trans_train_data, batch_size)
        test_x = divide_batches_gen(trans_test_data, batch_size)

        # Train Data.
        count = 0
        train_loss = 0

        for train_data, train_label in zip(train_x, train_y):
            train_count += 1
            count += 1

            _, l = sess.run([optimizer, loss], feed_dict={x: train_data, y: train_label})

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

        trans_train_label = trans_train_label.reshape(trans_train_label.shape[0], )
        train_predictions = np.asarray(train_predictions)
        train_predictions = train_predictions.reshape(train_predictions.shape[0], )

        trans_test_label = trans_test_label.reshape(trans_test_label.shape[0], )
        test_predictions = np.asarray(test_predictions)
        test_predictions = test_predictions.reshape(test_predictions.shape[0], )

        train_gini = calculate_gini_score_manual(np.asarray(train_predictions), trans_train_label)
        test_gini = calculate_gini_score_manual(np.asarray(train_predictions), trans_train_label)

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
        print("\n\n")

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

        model_saver.save(sess, '../maxlife_models/' + model_name + '/trans_model', global_step=i)
