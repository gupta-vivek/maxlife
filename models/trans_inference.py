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
from utils.data_utils import calculate_decile, calculate_gini_score_manual
import numpy as np
import os

# infer_path = "/Users/vivek/sample.csv"
infer_path = sys.argv[1]
model_name = sys.argv[2]

batch_size = 512
display_count = 1000
split_ratio = [100, 0, 0]

print("Reading the data...")
inference_data, inference_label, _, _, _, _ = read_csv(infer_path, split_ratio=split_ratio, header=True, ignore_cols=["POL_ID", "DATA_MONTH", "MODE_OF_PAYMENT", "MI"], output_label="Lapse_Flag")

print(inference_data[0])

print("Infer Data Size - ", len(inference_data))

print("Splitting the data...")
infer_y = divide_batches(inference_label, batch_size)

infer_batch_size = len(infer_y)

saved_model_dir ="../maxlife_models/"
if not os.path.isdir(saved_model_dir):
    os.mkdir(saved_model_dir)

with tf.device("/GPU:0"):
    with tf.Session() as sess:
        saved_model = saved_model_dir + 'trans_model/'
        ckpt = tf.train.latest_checkpoint(saved_model)
        filename = ".".join([ckpt, 'meta'])
        previous_count = int(filename.split('-')[1].split('.')[0])
        model_saver = tf.train.import_meta_graph(filename)
        model_saver.restore(sess, ckpt)

        graph = tf.get_default_graph()

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        x = graph.get_tensor_by_name('placeholders/input:0')
        y = graph.get_tensor_by_name('placeholders/output:0')
        z = graph.get_tensor_by_name('placeholders/z:0')

        y_ = tf.get_collection("y_")[0]
        loss = tf.get_collection("loss")[0]

        infer_x = divide_batches_gen(inference_data, batch_size)

        count = 0
        infer_loss = 0
        infer_predictions = []

        for infer_data, infer_label in zip(infer_x, infer_y):
            count += 1

            l, model_prediction = sess.run([loss, y_], feed_dict={x: infer_data, y: infer_label})

            infer_loss += l
            infer_predictions.append(temp for temp in model_prediction)
            if count % display_count == 0:
                print("Test Batch Count: ", count)
                print("Test Iter Loss: ", l)

        infer_predictions = [item for sublist in infer_predictions for item in sublist]

        print("infer pred - ", len(infer_predictions))
        print("infer label - ", len(list(inference_label)))

        infer_decile_score = calculate_decile(infer_predictions, list(inference_label))

        inference_label = inference_label.reshape(inference_label.shape[0], )
        infer_predictions = np.asarray(infer_predictions)
        infer_predictions = infer_predictions.reshape(infer_predictions.shape[0], )

        infer_gini = calculate_gini_score_manual(np.asarray(infer_predictions), inference_label)

        print("Inference")
        print("Loss: ", infer_loss / infer_batch_size)
        print("Decile: ", infer_decile_score)
        print("Gini: ", infer_gini)
        print("\n\n")