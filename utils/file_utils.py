import numpy as np
import pandas as pd
import logging
from typing import Union
from ast import literal_eval
from typeguard import typechecked

logger = logging.getLogger(__name__)


def read_csv(filename: Union[str, object] = None, data: Union[list, np.ndarray, pd.DataFrame] = None,
             delimiter: str = ',', normalize: bool = False,
             dtype=None, split_ratio: Union[list, None] = None,
             header: Union[bool, int, list] = None, skiprows: int = None, ignore_cols=None,
             select_cols: list = None,
             index_col: int = False, output_label: Union[str, int, list, bool] = True, randomize: bool = False,
             describe: bool = False, label_vector: bool = False):
    """
    | Function to read a csv file.
    |
    :param filename: File name with absolute path
    :param data: Data used for train and test
    :param delimiter: Delimiter used to split columns
    :param normalize: Normalize the Data
    :param dtype: Data Format
    :param split_ratio: Ratio used to split the data.
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param ignore_cols: Ignore specified columns
    :param select_cols: Selects only specified columns
    :param index_col: Index Column
    :param output_label: Column which specifies whether output label should be available or not.
    :param randomize: Randomize data
    :param describe: Describe Input Data
    :param label_vector: True if output label is a vector
    :return: dataframe
    """

    header = 0 if header else None
    if filename:
        df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype,
                         skiprows=skiprows)
    elif isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise Exception('Filename / Data are None. Specify atleast one source')
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    df = df.apply(lambda x: np.log(x)) if normalize else df

    if output_label is None or output_label is False:
        if ignore_cols:
            ignore_cols = range_cols(ignore_cols)
            df = df.drop(ignore_cols, axis=1)
        if select_cols:
            select_cols = range_cols(select_cols)
            df = df[select_cols]
        if split_ratio is not None:
            df = split(df, split_ratio)

        return df

    elif output_label is not None:
        if header is None:
            if output_label is True:
                column_drop = len(df.columns) - 1
            else:
                if len(output_label[0]) > 1:
                    column_drop = range_cols(output_label)
                else:
                    column_drop = [output_label[0][0] - 1]
            if ignore_cols:
                ignore_cols = range_cols(ignore_cols)
                df = df.drop(ignore_cols, axis=1)
            label_df = df[column_drop].apply(literal_eval) if label_vector else df[
                column_drop]
            df = df.drop(column_drop, axis=1)
            if select_cols:
                select_cols = range_cols(select_cols)
                df = df[select_cols]
        else:
            column_drop = df.columns[-1] if output_label is True else output_label
            if ignore_cols:
                df = df.drop(ignore_cols, axis=1)
            label_df = df[column_drop].apply(literal_eval) if label_vector else df[
                column_drop]
            df = df.drop(column_drop, axis=1)

            if select_cols:
                df = df[select_cols]

        if split_ratio is not None:
            df = split(df, split_ratio)
            label_df = split(label_df, split_ratio)

        return df, label_df


@staticmethod
def convert_df_tolist(*input_data):
    """
    | This function converts df to list.
    |
    :param input_data: Input Data (*args)
    :return: dataframe
    """
    dataframes = []
    for df in input_data:
        if len(df) == 0:
            continue
        else:
            if isinstance(df, pd.DataFrame):
                if len(input_data) == 1:
                    return df.values.tolist()
                dataframes.append(df.values.tolist())
            elif isinstance(df, pd.Series):
                df_list = df.to_frame().values.tolist()
                if isinstance(df_list, list):
                    if isinstance(df_list[0][0], list):
                        dataframes.append([i[0] for i in df.to_frame().values.tolist()])
                    else:
                        dataframes.append(df.to_frame().values.tolist())
                else:
                    dataframes.append(df.to_frame().values.tolist())
    return dataframes


def split( df, split_ratio):
    """
    | This functions splits the dataframe in the given ratio.
    |
    :param df: Dataframe
    :param split_ratio: Ratio used to split data
    :return: split data
    """
    if sum(split_ratio) == 100:
        length = len(split_ratio)
        data = []
        start_size = 0
        for i in range(length):
            final_size = (int(len(df) * split_ratio[i] / 100))
            data.append(df.iloc[start_size:start_size + final_size])
            start_size += final_size
        return data
    else:
        raise Exception("Sum of elements should be equal to 100!")


@typechecked
def range_cols(cols: list = None):
    """
    | This functions returns the range of columns.
    |
    :param cols: list of columns
    :return: range of columns
    """
    if isinstance(cols[0], list):
        col_range = []
        for each_range in cols:
            col_range += list(range(each_range[0], each_range[1]))

    return col_range


@typechecked
def to_csv(data: Union[dict, list, np.ndarray], save_path: str, header: bool = True, index: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | Write to CSV File
    :param data: Data
    :param save_path: Save Path
    :param header: Header
    :param index: Index
    | ..todo ::
        Prathyush SP:
        1. Fix Logging Issue
    """
    if isinstance(data, dict):
        keys, vals = [], []
        for k, v in data.items():
            keys.append(k)
            vals.append(v)
        vals = list(zip(*vals))
        pd.DataFrame(data=vals, columns=keys).to_csv(path_or_buf=save_path, header=header, index=index)
    else:
        pd.DataFrame(data=data).to_csv(path_or_buf=save_path, header=header, index=index)
    logger.info('CSV Generated at ' + save_path)
    print('CSV Generated at', save_path)