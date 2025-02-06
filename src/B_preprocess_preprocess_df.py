"""
Normalized the dataframe.
Features will be scaled to [0,1]
用于将df进行归一化，可以用，但先有rawdata的格式

"""

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series, read_pickle, set_option
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler

from utils_env import training_columns_regex
from utils_file_saver import get_decorated_filepath, save_figure, save_obj
from utils_paths import PATH_DATAFRAME


def df_filter_columns_by_std(X_train: Series, X_test: Series, std=0.01):
    """
    根据标准差过滤列。
    移除标准差小于等于 0.1 的列。
    
    参数:
    - X_train: 训练数据集
    - X_test: 测试数据集
    - std: 标准差阈值，默认为 0.01
    
    返回:
    - 过滤后的训练数据集和测试数据集
    """
    X_test = X_test.loc[:, X_train.std() > 0.1]
    X_train = X_train.loc[:, X_train.std() > 0.1]
    return X_train, X_test


def split_and_normalize(X: Series, y: Series, test_size: float, columns_to_scale, scaler: MinMaxScaler = MinMaxScaler()):
    """
    将数据集分割成训练集和测试集，并对指定的列进行归一化。
    归一化范围为 [0, 1]。
    
    参数:
    - X: 特征数据集
    - y: 标签数据集
    - test_size: 测试集占总数据的比例
    - columns_to_scale: 需要归一化的列名列表或布尔列表
    - scaler: 归一化工具，默认为 MinMaxScaler
    
    返回:
    - 分割并归一化后的训练集、测试集、训练标签和测试标签
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # 对指定的列进行归一化
    X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train.loc[:, columns_to_scale])
    X_test.loc[:, columns_to_scale] = scaler.transform(X_test.loc[:, columns_to_scale])
    return X_train, X_test, y_train, y_test


def df_replace_values(df: DataFrame):
    """
    替换数据框中的无效值并进行初步的标准化处理。
    将无穷大（inf）和负无穷大（-inf）替换为 NaN，并将 NaN 填充为 0。
    
    参数:
    - df: 输入的数据框
    
    返回:
    - 处理后的数据框
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    # 设置Pandas显示选项，显示所有列
    set_option("display.max_columns", None)
    # 忽略警告信息
    warnings.filterwarnings("ignore")

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入的数据框路径
    parser.add_argument("--df", metavar="df", type=str, help="加载预先计算的熵数据框（该数据框尚未清理和归一化）")
    # 添加命令行参数：输出目录路径，默认为 PATH_DATAFRAME
    parser.add_argument("--output-dir", metavar="dir", type=str, help="保存数据框和npy文件的目录", default=PATH_DATAFRAME)
    # 解析命令行参数
    args = parser.parse_args()

    # 获取数据框路径
    df_path = Path(args.df)
    # 获取输出目录路径
    output_dir = Path(args.output_dir)

    # 加载数据框
    df: DataFrame = read_pickle(df_path)
    
    # 提取训练列，使用正则表达式匹配
    training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    
    # 替换无穷大和负无穷大为 NaN，并将 NaN 填充为 0
    df = df_replace_values(df)

    # 生成新的文件名，将 "raw" 替换为 "cleaned2"
    basename = df_path.stem.replace("raw", "cleaned2")
    
    # 定义文件保存函数
    file_saver = lambda df, filepath: DataFrame.to_pickle(df, filepath)
    
    # 获取装饰后的文件路径
    filepath = get_decorated_filepath(directory=output_dir, basename=basename, extension=".pkl")
    
    # 保存清理后的数据框
    save_obj(obj=df, filepath=filepath, file_saver=file_saver, metadata={})



