# -*- coding: utf-8 -*-
"""
@created on: 8/21/18,
@author: Vivek A Gupta,

Description:

..todo::
"""
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.special import expit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle


def sigmoid(z):
    return expit(z)


def calculate_gini_score_manual(y_true, y_pred):
    """
    | **@author**: Prathyush SP
    |
    | Calculate Gini Score
    :param y_true: Actual Label
    :param y_pred: Predicted Label
    :return: Gini Score
    """

    y_pred = sigmoid(y_pred)
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    arr = np.array([y_true, y_pred]).transpose()
    true_order, pred_order = arr[arr[:, 0].argsort()][::-1, 0], arr[arr[:, 1].argsort()][::-1, 0]
    l_true, l_pred, l_ones = np.cumsum(true_order) / np.sum(true_order), np.cumsum(pred_order) / np.sum(
        pred_order), np.linspace(1 / n_samples, 1, n_samples)
    g_true, g_pred = np.sum(l_ones - l_true), np.sum(l_ones - l_pred)
    return g_pred / g_true


def calculate_decile(predictions, labels):
    sum_ones = np.count_nonzero(labels)
    sum_zeroes = len(labels) - sum_ones

    decile_dict = OrderedDict()

    df = pd.DataFrame()
    df['Label'] = labels

    preds_sig = sigmoid(predictions)

    df['Predictions'] = preds_sig
    df = df.sort_values(by=['Predictions'], ascending=False)

    length_predictions = len(preds_sig)
    length_predictions_10 = int(length_predictions/10)
    length_predictions_10_x = length_predictions_10

    for i in reversed(range(10)):
        decile_dict[str(i)] = {}
        decile_dict[str(i)]['zero'] = 0
        decile_dict[str(i)]['one'] = 0

    x = 0
    for i in reversed(range(10)):
        temp_df = df[x:length_predictions_10]
        sum_of_one = np.count_nonzero(temp_df['Label'])
        sum_of_zero = length_predictions_10_x - sum_of_one
        decile_dict[str(i)]['zero'] = sum_of_zero
        decile_dict[str(i)]['one'] = sum_of_one

        if sum_of_one == 0:
            percentage_ones = 0
        else:
            percentage_ones = decile_dict[str(i)]['one'] / sum_ones * 100

        if sum_of_zero == 0:
            percentage_zeroes = 0
        else:
            percentage_zeroes = decile_dict[str(i)]['zero'] / sum_zeroes * 100

        decile_dict[str(i)]['percentage_ones'] = percentage_ones
        decile_dict[str(i)]['percentage_zeroes'] = percentage_zeroes

        x = length_predictions_10
        length_predictions_10 += length_predictions_10_x

    dec_score = 0
    count = 0
    for k in decile_dict.keys():
        count += 1
        if count <= 3:
            dec_score += decile_dict[k]['percentage_ones']
        else:
            break
    return dec_score


def stratified_generator(trans_path, train_ratio, test_ratio, output_dir):
    trans_df = pd.read_csv(trans_path)
    trans_df_Y = trans_df[['Lapse_Flag', 'TB_POL_BILL_MODE_CD']]
    trans_df_X = trans_df.drop(['Lapse_Flag', 'TB_POL_BILL_MODE_CD'], axis=1)

    train_ratio = float(train_ratio)
    test_ratio = float(test_ratio)

    print("Stratified split...")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=test_ratio)

    # trans_df['Lapse_Flag'] = trans_df['Lapse_Flag']
    # trans_df = trans_df.drop(['Lapse_Flag'], axis=1)

    train_index = None
    test_index = None

    for train_ind, test_ind in sss.split(trans_df_X, trans_df_Y):
        train_index = train_ind
        test_index = test_ind

    print("Train")
    print("Transaction")
    train_df = trans_df.reindex(train_index)
    train_df.to_csv(output_dir + "final_trans_train.csv", index=False)
    print(train_df.head())

    print("Test")
    print("Transaction")
    test_df = trans_df.reindex(test_index)
    test_df.to_csv(output_dir + "final_trans_test.csv", index=False)
    print(test_df.head())


def data_generator(data_path, ratio_ones=0.3, ratio_zeroes=0.7, length=2100000, output_dir=None, test=True):
    length = int(length)
    ratio_zeroes = float(ratio_zeroes)
    ratio_ones = float(ratio_ones)

    df = pd.read_csv(data_path)
    print(len(df))
    one_df = df.loc[df['Lapse_Flag'] == 1.]
    one_df = shuffle(one_df)
    zero_df = df.loc[df['Lapse_Flag'] == 0.]
    zero_df = shuffle(zero_df)

    count_one = int(ratio_ones * length)
    count_zero = int(ratio_zeroes * length)
    new_one_df = one_df[:count_one]
    new_one_df_test = one_df[count_one:]
    new_zero_df = zero_df[:count_zero]
    new_zero_df_test = zero_df[count_zero:]

    new_df = pd.concat([new_one_df, new_zero_df])
    # new_df = new_df.drop(['POL_ID'], axis=1)
    new_df = shuffle(new_df)
    new_df.to_csv(output_dir + "/trans2_train.csv", index=False)

    if test:
        new_df_test = pd.concat([new_one_df_test, new_zero_df_test])
        # new_df_test = new_df_test.drop(['POL_ID'], axis=1)
        new_df_test['Lapse_Flag'] = new_df_test['Lapse_Flag']
        new_df_test = shuffle(new_df_test)
        new_df_test.to_csv(output_dir + "/trans2_test.csv", index=False)


# data_generator("/Users/vivek/transaction.csv", output_dir="/Users/vivek/")
# stratified_generator("/Users/vivek/final_trans.csv", train_ratio=0.5, test_ratio=0.5, output_dir="/Users/vivek/")
