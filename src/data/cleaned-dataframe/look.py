import pandas as pd
import pickle
import os

# 指定 pkl 文件的路径
pkl_file_path = 'D:\Code\EEG_paper_2025\MEFA\eeg-driver-fatigue-detection\data\dataframes\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle'

# 读取 pkl 文件
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 检查是否是 DataFrame
if isinstance(data, pd.DataFrame):
    print("文件内容是一个 DataFrame")
    print("\n数据的前几行:")
    print(data.head())

    # 导出前5行到CSV文件
    output_dir = os.path.dirname(pkl_file_path)
    output_file = os.path.join(output_dir, 'first_5_rows.csv')
    data.head().to_csv(output_file, index=False)
    print(f"\n前5行已导出到: {output_file}")
    print("\nDataFrame 的基本信息:")
    print(data.info())

    print("\n数据的统计摘要:")
    print(data.describe())
else:
    print("文件内容不是 DataFrame")
    print("内容类型:", type(data))
