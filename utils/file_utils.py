# -*- coding: utf-8 -*-
"""
@created on: 8/20/18,
@author: Vivek A Gupta,

Description:

..todo::
"""

import logging
from ast import literal_eval
from typing import Union
import numpy as np
import pandas as pd

from typeguard import typechecked

logger = logging.getLogger(__name__)


def divide_batches_gen(input_batch, batch_size):
    """
    Divide into batches

    :param input_batch:
    :param batch_size:
    :return:
    """
    output_batch = []
    for i in range(0, len(input_batch), batch_size):
        # output_batch.append(input_batch[i: i + batch_size])
        yield input_batch[i: i + batch_size]
    # return output_batch
    # yield output_batch


def divide_batches(input_batch, batch_size):
    """
    Divide into batches

    :param input_batch:
    :param batch_size:
    :return:
    """
    output_batch = []
    for i in range(0, len(input_batch), batch_size):
        output_batch.append(input_batch[i: i + batch_size])
    return output_batch


def convert_df_tolist(*input_data, strict_type):
    """
    | **@author:** Prathyush SP
    |
    | Convert Dataframe to List
    |
    :param strict_type: Covert to default data type
    :param input_data: Input Data (*args)
    :return: Dataframe
    .. todo::
        Prathyush SP:
            1. Check list logic
            2. Perform Dataframe Validation
    """
    dataframes = []
    for df in input_data:
        # if strict_type:
        #     df = df.astype(RZTDL_CONFIG.TensorflowConfig.DTYPE.as_numpy_dtype)
        if isinstance(df, pd.DataFrame):
            if len(input_data) == 1:
                return df.values
            dataframes.append(df.values)
        elif isinstance(df, pd.Series):
            df_list = df.to_frame().values
            if isinstance(df_list, list):
                if isinstance(df_list[0][0], list):
                    dataframes.append([i[0] for i in df.to_frame().values])
                else:
                    dataframes.append(df.to_frame().values)
            else:
                dataframes.append(df.to_frame().values)
        elif isinstance(df, np.ndarray):
            dataframes.append(df)
    return dataframes


@typechecked
def read_csv(filename: Union[str, object] = None, data: Union[list, np.ndarray, pd.DataFrame] = None,
             split_ratio: Union[list, None] = (100, 0, 0), delimiter: str = ',', dtype=None,
             header: Union[bool, int, list] = None, skiprows: int = None, ignore_cols=None, select_cols: list = None,
             index_col: Union[int, None] = False, output_label: Union[str, int, list, bool] = True,
             randomize: bool = False, return_as_dataframe: bool = False, describe: bool = False,
             label_vector: bool = False, strict_type: bool = True):
    """
    | **@author:** Prathyush SP
    |
    | The function is used to read a csv file with a specified delimiter
    :param select_cols: select particular columns to train the data
    :param ignore_cols: ignore particular columns for training
    :param strict_type: Converting data to float type
    :param filename: File name with absolute path
    :param data: Data used for train and test
    :param split_ratio: Ratio used to split data into train and test
    :param delimiter: Delimiter used to split columns
    :param dtype: Data Format
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param index_col: Index Column
    :param output_label: Column which specifies whether output label should be available or not.
    :param randomize: Randomize data
    :param return_as_dataframe: Returns as a dataframes
    :param describe: Describe Input Data
    :param label_vector: True if output label is a vector
    :return: return train_data, train_label, test_data, test_label based on return_as_dataframe
    """
    header = 0 if header else None
    if filename:
        df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype, skiprows=skiprows)
    elif isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise Exception('Filename / Data are None. Specify atleast one source')
    index_column = df.index if index_col is not False and index_col is not None else []
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    if len(split_ratio) == 3 and sum(split_ratio) == 100:
        test_size = int(len(df) * split_ratio[-1] / 100)
        valid_size = int(len(df) * split_ratio[1] / 100)
        train_size = int(len(df) - (test_size + valid_size))
        train_data_df, valid_data_df, test_data_df = df.head(int(train_size)), \
                                                     df.iloc[train_size:(train_size + valid_size)], \
                                                     df.tail(int(test_size))
        if output_label is None or output_label is False:
            if ignore_cols:
                ignore_cols = select_range_of_columns(ignore_cols)
                train_data_df = train_data_df.drop(ignore_cols, axis=1)
                test_data_df = test_data_df.drop(ignore_cols, axis=1)
                valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
            if select_cols:
                select_cols = select_range_of_columns(select_cols)
                train_data_df = train_data_df[select_cols]
                test_data_df = test_data_df[select_cols]
                valid_data_df = valid_data_df[select_cols]
            if return_as_dataframe:
                return train_data_df, valid_data_df, test_data_df
            elif index_col is not False and index_col is not None:
                return convert_df_tolist(index_column.values, train_data_df, valid_data_df, test_data_df,
                                         strict_type=strict_type)
            else:
                return convert_df_tolist(train_data_df, valid_data_df, test_data_df, strict_type=strict_type)
        elif output_label is not None:
            if header is None:
                if output_label is True:
                    column_drop = len(
                        train_data_df.columns) if index_col is not False and index_col is not None else len(
                        train_data_df.columns) - 1
                else:
                    column_drop = select_range_of_columns(output_label)
                if ignore_cols:
                    ignore_cols = select_range_of_columns(ignore_cols)
                    train_data_df = train_data_df.drop(ignore_cols, axis=1)
                    test_data_df = test_data_df.drop(ignore_cols, axis=1)
                    valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
                train_label_df = np.array(
                    [arr for arr in train_data_df[column_drop].apply(literal_eval)]) if label_vector else train_data_df[
                    column_drop]
                train_data_df = train_data_df.drop(column_drop, axis=1)
                valid_label_df = np.array(
                    [arr for arr in valid_data_df[column_drop].apply(literal_eval)]) if label_vector else valid_data_df[
                    column_drop]
                valid_data_df = valid_data_df.drop(column_drop, axis=1)
                test_label_df = np.array(
                    [arr for arr in test_data_df[column_drop].apply(literal_eval)]) if label_vector else test_data_df[
                    column_drop]
                test_data_df = test_data_df.drop(column_drop, axis=1)
                if select_cols:
                    select_cols = select_range_of_columns(select_cols)
                    train_data_df = train_data_df[select_cols]
                    test_data_df = test_data_df[select_cols]
                    valid_data_df = valid_data_df[select_cols]
            else:
                column_drop = df.columns[-1] if output_label is True else output_label
                if ignore_cols:
                    ignore_cols = select_range_of_columns(ignore_cols)
                    train_data_df = train_data_df.drop(ignore_cols, axis=1)
                    test_data_df = test_data_df.drop(ignore_cols, axis=1)
                    valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
                train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[
                    column_drop]
                train_data_df = train_data_df.drop(column_drop, axis=1)
                valid_label_df = valid_data_df[column_drop].apply(literal_eval) if label_vector else valid_data_df[
                    column_drop]
                valid_data_df = valid_data_df.drop(column_drop, axis=1)
                test_label_df = test_data_df[column_drop].apply(literal_eval) if label_vector else test_data_df[
                    column_drop]
                test_data_df = test_data_df.drop(column_drop, axis=1)
                if select_cols:
                    select_cols = select_range_of_columns(select_cols)
                    train_data_df = train_data_df[select_cols]
                    test_data_df = test_data_df[select_cols]
                    valid_data_df = valid_data_df[select_cols]
            if return_as_dataframe:
                return train_data_df, train_label_df, valid_data_df, valid_label_df, test_data_df, test_label_df
            elif index_col is not False and index_col is not None:
                return convert_df_tolist(index_column.values, train_data_df, train_label_df, valid_data_df,
                                         valid_label_df, test_data_df, test_label_df, strict_type=strict_type)
            else:
                return convert_df_tolist(train_data_df, train_label_df, valid_data_df,
                                         valid_label_df, test_data_df, test_label_df, strict_type=strict_type)
    else:
        raise Exception("Length of split_ratio should be 3 with sum of elements equal to 100")


@typechecked
def select_range_of_columns(select_cols: list = None):
    if isinstance(select_cols[0], str):
        return select_cols
    else:
        if isinstance(select_cols[0], list):
            col_range = []
            for each_range in select_cols:
                if len(each_range) != 2:
                    raise Exception("Select the column range with two values")
                col_range += list(range(each_range[0], each_range[1]))
            return col_range
        return select_cols


@typechecked
def to_csv(index_col: Union[list, np.ndarray], result: Union[list, np.ndarray], save_path: str,
           header: bool = True, index: bool = False):
    """
    | **@author:** Umesh Kumar
    |
    | Write to CSV File

    :param index_col: Index column for dataset
    :param result: Predicted Result
    :param save_path: Save path where to save
    :param header: Header
    :param index: Index
    :return:
    """
    shape = result[0].shape
    flatten_shape, flattened_values = shape[1:], []
    for i in range(len(result[0])):
        flattened_values.append(result[0][i].flatten().tolist())

    flatten_shape = np.ndarray(flatten_shape)
    columns, column_dimension = ["id"], []

    result = [[index] for index in index_col]

    for i, value in enumerate(flattened_values):
        result[i] = result[i] + value

    get_column_name(flatten_shape=flatten_shape, column_dimension=column_dimension, columns=columns)
    dataframe = pd.DataFrame(result, columns=columns)
    dataframe.to_csv(path_or_buf=save_path, header=header, index=index)
    logger.info("Filed saved successfully in " + save_path)


@typechecked
def get_column_name(flatten_shape: Union[np.ndarray, np.float64], column_dimension: list, columns: list):
    """
    | **@author:** Umesh Kumar
    |

    :param columns:
    :param flatten_shape:
    :param column_dimension:
    :return:
    """
    if isinstance(flatten_shape, np.ndarray):
        for i in range(len(flatten_shape)):
            column_dimension.append("_" + str(i))
            get_column_name(flatten_shape[i], column_dimension, columns=columns)
            column_dimension.pop()
    else:
        col_dimension = ''.join(column_dimension)
        col_dimension = col_dimension[1:]
        columns.append("output_" + col_dimension)
        return