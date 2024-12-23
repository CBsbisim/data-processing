import ray
import pandas as pd
from models.data_processing import DP
import asyncio
from pyhive import hive
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
# Connect to distributed database
# CONN = hive.Connection(host="172.18.0.3", port=10000, username="hdfs")
# CURSOR = CONN.cursor()

# Time range for training and trading data
TRAIN_START_DATE = "2020-03-01"
TRAIN_END_DATE = "2022-6-30"
TRADE_START_DATE = "2022-6-30"
TRADE_END_DATE = "2023-6-30"

# Initialize Ray with more CPUs and GPUs if needed
ray.init(num_cpus=18, num_gpus=0)

# Define remote tasks
@ray.remote
def fetch_data(func, start_date, end_date):
    try:
        print(f"Fetching data from {func.__name__}...")
        data = func(start_date=start_date, end_date=end_date)
        print(f"Completed {func.__name__}.")
        return data
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return pd.DataFrame()

# 数据分割任务（并行处理）
@ray.remote
def split_data(dp,processed_full, is_train=True):
    data = dp.data_split(processed_full, TRAIN_START_DATE if is_train else TRADE_START_DATE, TRAIN_END_DATE if is_train else TRADE_END_DATE)
    data = data.sort_values(by='date', ascending=True)
    data['0'] = pd.factorize(data['date'])[0]
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.tz_localize('Asia/Shanghai')
    data['day'] = data['date'].dt.weekday
    data['date'] = data['date'].dt.date
    # 去掉周末数据
    data = data[(data['day'] != 6) & (data['day'] != 5)]
    return data

def clean(data):  # Pass AK and YF as parameters if they're pre-loaded
    dirt = []
    for i in set(data.date.values):
        ifproper = data[data['date']==i]['close'].size
        if ifproper != data.tic.unique().size:
            dirt.append(i)
    data['date'] = pd.to_datetime(data['date'])
    dates_to_remove = pd.to_datetime(dirt)
    if dates_to_remove.size>0:
        data = data[~data['date'].isin(dates_to_remove)]
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.date
    return data


def main(ak_list,yf_list,bond_list):

    dp = DP(ak_list,yf_list,bond_list)
    dp.get_market_situation()
    
    # List of functions and their arguments
    tasks = [
        (dp.get_stock_data_from_ak, "20150101", "20240131"),
        (dp.get_stock_data_from_yf, "2015-01-01", "2024-01-31"),
        (dp.get_bond_data_from_ak, "2015-01-01", "2024-01-31"),
    ]
    
    # Launch all fetch tasks asynchronously
    futures = [fetch_data.remote(func, start, end) for func, start, end in tasks]

    # Collect all results in parallel
    try:
        data_results = ray.get(futures)
        print("All fectching tasks completed.")
    except Exception as e:
        print(f"Error during task execution: {e}")
        data_results = []

    # 合并债券和基金数据
    stock_data_ak, stock_data_yf,bond_data = data_results
    merged_data = pd.concat([stock_data_ak, stock_data_yf,bond_data], axis=0)
    merged_data = merged_data.drop_duplicates()

    # 清洗 数据
    try:
        cleaned_data = clean(merged_data)
        print("Data cleaninging completed.")
    except Exception as e:
        print(f"Error during task execution: {e}")
        cleaned_data = []

    # 分割数据
    t_futures = [split_data.remote(dp,cleaned_data, True),split_data.remote(dp,cleaned_data, False)]
    # 获取分割后的数据
    try:
        train_data,trade_data = ray.get(t_futures)
        print("Data splitting completed.")
    except Exception as e:
        print(f"Error during task execution: {e}")
        train_data,trade_data = [],[]


    # 保存合并后的数据
    train_data.to_csv("./his_data/train_data.csv", index=False)
    trade_data.to_csv("./his_data/trade_data.csv", index=False)
    # 保上传合并后的数据
    # upload_data("train_data",train_data)
    # upload_data("trade_data",trade_data)
    print("Train data and trade data prepared successfully!")
    print(f"Train data size: {train_data.shape}")
    print(f"Trade data size: {trade_data.shape}")

if __name__ == "__main__":
    print("ok")
    
    