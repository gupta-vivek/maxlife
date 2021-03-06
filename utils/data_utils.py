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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def sigmoid(z):
    return expit(z)


def calculate_precision_recall_curve(y_true, y_pred):
    y_pred = sigmoid(y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return precision, recall, thresholds


def calculate_confusion_matrix(y_true, y_pred, threshold):
    y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred >= threshold)] = 1.0
    y_pred[np.where(y_pred < threshold)] = 0.0
    return confusion_matrix(y_true, y_pred)


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


def calculate_decile(predictions, labels, decile_distribution=False):
    sum_ones = np.count_nonzero(labels)
    sum_zeroes = len(labels) - sum_ones

    decile_dict = OrderedDict()

    df = pd.DataFrame()
    df['Label'] = labels

    preds_sig = sigmoid(predictions)

    df['Predictions'] = preds_sig
    df = df.sort_values(by=['Predictions'], ascending=False)

    length_predictions = len(preds_sig)
    length_predictions_10 = int(length_predictions / 10)
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

    if decile_distribution:
        print("Decile Distribution")
        print(decile_dict)

    dec_score = 0
    count = 0
    for k in decile_dict.keys():
        count += 1
        if count <= 3:
            dec_score += decile_dict[k]['percentage_ones']
        else:
            break
    return dec_score


def calculate_decile_softmax(predictions, labels, decile_distribution=False):
    labels = np.logical_not(labels[:, -1]).astype(int)
    print(labels)
    sum_ones = np.count_nonzero(labels)
    sum_zeroes = labels.shape[0] - sum_ones

    decile_dict = OrderedDict()

    df = pd.DataFrame()
    new_prediction = np.max(predictions[:,:3], axis=1)
    print(new_prediction)

    df['Label'] = labels
    # preds_sig = sigmoid(predictions)
    preds_sig = new_prediction

    df['Predictions'] = preds_sig
    df = df.sort_values(by=['Predictions'], ascending=False)

    length_predictions = preds_sig.shape[0]
    length_predictions_10 = int(length_predictions / 10)
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

    if decile_distribution:
        print("Decile Distribution")
        print(decile_dict)

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
    trans_df_Y = trans_df[['Lapse_Flag']]
    trans_df_X = trans_df.drop(['Lapse_Flag'], axis=1)

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
    train_df.to_csv(output_dir + "half_train.csv", index=False)
    print(train_df.head())

    print("Test")
    print("Transaction")
    test_df = trans_df.reindex(test_index)
    test_df.to_csv(output_dir + "half_test.csv", index=False)
    print(test_df.head())


def stratified_generator2(ffn_path, lstm_path, train_ratio, test_ratio, output_dir):
    ffn_filename = ffn_path.split('/')[-1]
    ffn_filename = ffn_filename.split('.')[0]
    ffn_train_filename = ffn_filename + "_train.csv"
    ffn_test_filename = ffn_filename + "_test.csv"

    lstm_filename = lstm_path.split('/')[-1]
    lstm_filename = lstm_filename.split('.')[0]
    lstm_train_filename = lstm_filename + "_train.csv"
    lstm_test_filename = lstm_filename + "_test.csv"

    trans_df = pd.read_csv(lstm_path)
    trans_df_Y = trans_df[['Lapse_Flag']]
    trans_df_X = trans_df.drop(['Lapse_Flag'], axis=1)

    ffn_df = pd.read_csv(ffn_path)

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
    # train_df = trans_df.reindex(train_index)

    idx = np.random.permutation(train_index)
    train_trans_df = trans_df.reindex(idx)
    train_ffn_df = ffn_df.reindex(idx)

    train_trans_df.to_csv(output_dir + lstm_train_filename, index=False)
    print(train_trans_df.head())
    print(len(train_trans_df))

    print("FFN")
    train_ffn_df.to_csv(output_dir + ffn_train_filename, index=False)
    print(train_ffn_df.head())
    print(len(train_ffn_df))
    print("Test")
    print("Transaction")
    # test_df = trans_df.reindex(test_index)

    idx = np.random.permutation(test_index)
    test_trans_df = trans_df.reindex(idx)
    test_ffn_df = ffn_df.reindex(idx)

    test_trans_df.to_csv(output_dir + lstm_test_filename, index=False)
    print(test_trans_df.head())
    print(len(test_trans_df))
    print("FFN")
    test_ffn_df.to_csv(output_dir + ffn_test_filename, index=False)
    print(test_ffn_df.head())
    print(len(test_ffn_df))


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


def month_mode_split(ffn_path, lstm_path, output_dir):
    lstm_df = pd.read_csv(lstm_path)
    ffn_df = pd.read_csv(ffn_path)

    print("12")
    temp_lstm_df = lstm_df.loc[lstm_df["TB_POL_BILL_MODE_CD"] == 12]
    temp_ffn_df = ffn_df.reindex(temp_lstm_df.index)
    print(temp_lstm_df.head())
    print(temp_ffn_df.head())

    temp_lstm_df.to_csv(output_dir + "annual_lstm.csv", index=False)
    temp_ffn_df.to_csv(output_dir + "annual_ffn.csv", index=False)

    print("6")
    temp_lstm_df = lstm_df.loc[lstm_df["TB_POL_BILL_MODE_CD"] == 6]
    temp_ffn_df = ffn_df.reindex(temp_lstm_df.index)
    print(temp_lstm_df.head())
    print(temp_ffn_df.head())

    temp_lstm_df.to_csv(output_dir + "half_lstm.csv", index=False)
    temp_ffn_df.to_csv(output_dir + "half_ffn.csv", index=False)

    print("3")
    temp_lstm_df = lstm_df.loc[lstm_df["TB_POL_BILL_MODE_CD"] == 3]
    temp_ffn_df = ffn_df.reindex(temp_lstm_df.index)
    print(temp_lstm_df.head())
    print(temp_ffn_df.head())

    temp_lstm_df.to_csv(output_dir + "quart_lstm.csv", index=False)
    temp_ffn_df.to_csv(output_dir + "quart_ffn.csv", index=False)

    print("1")
    temp_lstm_df = lstm_df.loc[lstm_df["TB_POL_BILL_MODE_CD"] == 1]
    temp_ffn_df = ffn_df.reindex(temp_lstm_df.index)
    print(temp_lstm_df.head())
    print(temp_ffn_df.head())

    temp_lstm_df.to_csv(output_dir + "monthly_lstm.csv", index=False)
    temp_ffn_df.to_csv(output_dir + "monthly_ffn.csv", index=False)


if __name__ == "__main__":
    # stratified_generator2("~/yearly_ffn2.csv", "~/yearly_lstm2.csv", train_ratio=0.6, test_ratio=0.4, output_dir="~/")

    a = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.2], [0.5, 0.1, 0.2, 0.2]])
    b = np.asarray([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    # print(b.shape)
    print(calculate_decile_softmax(a, b))