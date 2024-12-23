import pandas as pd
import numpy as np
import akshare as ak
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import ta
import random
import yfinance as yf
from stable_baselines3.common.logger import configure
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
import itertools

# 数据转换函数
def converting(data):
    columns = list(data.columns)
    columns[0] = '日期'
    data.columns = columns
    # 确保 '日期' 列为字符串类型
    data['日期'] = data['日期'].astype(str)
    # 将日期格式化为标准格式
    data['日期'] = data['日期'].str.replace('年', '-', regex=True)
    data['日期'] = data['日期'].str.replace('月份', '-01', regex=True)  # 默认加上日为 01
    data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m-%d')  # 转为日期类型
    # 提取年和月并存入新的列
    data["year"] = data["日期"].dt.year
    data["month"] = data["日期"].dt.month
    data = data.drop(columns = ['日期'])
    return data

# 训练数据提取与处理
def data_process(df_raw,use_turbulence):
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = INDICATORS,
                         use_vix=False,
                         use_turbulence=use_turbulence,
                         user_defined_feature = False)
    processed = fe.preprocess_data(df_raw)
    list_ticker = processed["tic"].unique().tolist()
    # 假设 list_date 和 list_ticker 已经定义
    list_date = pd.date_range(processed['date'].min(), processed['date'].max()).strftime('%Y-%m-%d').tolist()
    combination = list(itertools.product(list_date, list_ticker))
    # 创建 DataFrame
    processed_full = pd.DataFrame(combination, columns=["date", "tic"])
    # 确保在 merge 时 'date' 列的类型一致
    processed_full['date'] = pd.to_datetime(processed_full['date'])
    processed_full['date'] = processed_full['date'].dt.tz_localize('Asia/Shanghai')
    # 合并
    processed_full = processed_full.merge(processed, on=["date", "tic"], how="left")
    # 如果需要进一步操作
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)
    return processed_full

class DP:
    def __init__(self, ak_list,yf_list,bond_list):
        # Inside data_preparing.py
        self.ticker_list_ak = ak_list
        self.ticker_list_yf = yf_list
        self.bond_list = bond_list
        self.kimputer = KNNImputer(n_neighbors=5)
        self.imputer = IterativeImputer(max_iter=100, random_state=0)
    
    # 宏观数据获取函数
    def get_market_situation(self):
        data1 = ak.macro_usa_cb_consumer_confidence()[['日期','今值']]
        data1.columns = ['日期','consumer_confidence_usa']
        data2 = ak.macro_china_xfzxx()[['月份','消费者信心指数-指数值']]
        data2.columns = ['日期','consumer_confidence_cn']
        data3 = ak.macro_china_cpi_monthly()[['日期','今值']]
        data3.columns = ['日期','cpi_cn']
        data4 = YahooDownloader(ticker_list = ["^VIX"], start_date="2016-01-01", end_date="2024-02-01").fetch_data()
        self.vix = data4[['date','close']]
        self.vix.columns = ['date','vix']
        self.vix['date'] = pd.to_datetime(self.vix['date'])
        data5 = ak.index_news_sentiment_scope()[['日期','市场情绪指数']]
        data5.columns = ['日期','news_sentiment_scope']
        datalist = [data1,data2,data3,data5]
        for i in range(len(datalist)):
            datalist[i]=converting(datalist[i])
        data = datalist[0].merge(datalist[1], on=["year", "month"], how="outer").merge(datalist[2], on=["year", "month"], how="outer").merge(datalist[3], on=["year", "month"], how="outer")
        market = data[['year','month','consumer_confidence_usa','consumer_confidence_cn','cpi_cn','news_sentiment_scope']]
        # 数据填充
        df_imputed = self.imputer.fit_transform(market)
        self.market_data = pd.DataFrame(df_imputed, columns=market.columns)
        
        return None
    
    # 合并微观数据与宏观数据
    def merged_with_market(self,data):
        # 提取年和月并存入新的列
        data['date'] = pd.to_datetime(data['date'])
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data = data.merge(self.market_data, on=["year", "month"], how="left")
        data = data.drop(columns=["year","month"])
        data = data.merge(self.vix, on=['date'], how="left")
        return data
    
    # 由于有些股票可能早期没有上市因此股票数据结构有很多缺失值,因此要确保数据结构具有一致性
    # 根据市场信息填充数据
    def robust_structure(self,processed_full):
        processed_full['date'] = pd.to_datetime(processed_full['date'])
        processed_full['date'] = processed_full['date'].dt.tz_localize(None)
        processed_full['date'] = processed_full['date'].dt.date
        processed_full = processed_full.sort_values(by='date', ascending=True)
        processed_full = processed_full.drop_duplicates()
        processed_full = processed_full.reset_index(drop=True)
        dates_df = pd.DataFrame(processed_full['date'].drop_duplicates().values)
        tickers_df = pd.DataFrame(processed_full['tic'].drop_duplicates().values)
        # Use more memory-efficient merge strategy (avoiding large intermediate DataFrames)
        merged_data = pd.merge(dates_df, tickers_df, how='cross')  # Cross join more efficient
        merged_data.columns = ['date','tic']
        processed_full['date'] = pd.to_datetime(processed_full['date'])
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        merged_data = merged_data.merge(processed_full, on=['date','tic'],how='left')
        merged_data = self.merged_with_market(merged_data)
        if True in merged_data.isna().any().drop_duplicates().to_list():
            values = ['open', 'high', 'low', 'close', 'volume','macd','boll_ub', 'boll_lb', 'turbulence','rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma','consumer_confidence_usa','consumer_confidence_cn','cpi_cn','news_sentiment_scope','vix']
            df_imputed = self.imputer.fit_transform(merged_data[values])
            merged_data[values] = pd.DataFrame(df_imputed, columns=values).abs()
        return merged_data
    
    ## 提取某个指数下的股票数据
    # 为在中国的用户提供股票数据
    def get_data_from_akindex(self,start_date, end_date):
        ohlcv_data = []
        failed_tickers = []
        for ticker in self.ticker_list_ak:
            print(f"Fetching data for {ticker}...")
            success = False
            for _ in range(3):  # Retry 3 times
                try:
                    data = ak.stock_zh_a_hist(symbol=ticker[2:], period="daily", start_date=start_date, end_date=end_date)
                    if data.empty: raise ValueError(f"No data for {ticker}.")
                    data.columns = ["date", "tic", "open", "close", "high", "low", "volume", "day", "振幅", "涨跌幅", "涨跌额", "换手率"]
                    data['date'] = pd.to_datetime(data['date'])
                    data['date'] = data['date'].dt.tz_localize('Asia/Shanghai')
                    data['day'] = data['date'].dt.weekday
                    data = data[["date", "open","high", "low",  "close", "volume", "tic",'day']]
                    try:
                        data = data_process(data,True)
                    except Exception as e:
                        data = data_process(data,False)
                    ohlcv_data.append(data)
                    success = True
                    break
                except Exception as e:
                    print(f"Error: {e}")
            if not success:
                failed_tickers.append(ticker)
        if ohlcv_data:
            combined_data = pd.concat(ohlcv_data)
            return combined_data
        else:
            print(f"Failed to fetch data for: {', '.join(failed_tickers)}")
            return None
        
    def get_stock_data_from_ak(self,start_date, end_date):
        data = self.get_data_from_akindex(start_date, end_date)
        data = self.robust_structure(data)
        data['type'] = 1
        data.to_csv("./his_data/stock_data_from_ak.csv")
        return data
    
    # 获取所有沪深可转载数据
    def get_bond_data_from_ak(self,start_date, end_date):
        try:
            # 获取所有债券信息
            print("Fetching bond data...")
            all_bond_data = []
            for bond_code in self.bond_list:
                print(f"Fetching data for convertible bond: {bond_code}...")
                try:
                    bond_data = ak.bond_zh_hs_daily(symbol=bond_code)
                    bond_data['date'] = pd.to_datetime(bond_data['date'])  # 转换日期列
                    bond_data = bond_data[
                        (bond_data['date'] >= pd.to_datetime(start_date)) &
                        (bond_data['date'] <= pd.to_datetime(end_date))
                    ]
                    bond_data['tic'] = bond_code
                    bond_data['date'] = bond_data['date'].dt.tz_localize('Asia/Shanghai')
                    bond_data['day'] = bond_data['date'].dt.weekday
                    # 选择需要的列
                    bond_data = bond_data[['date', 'open', 'close', 'high', 'low', 'volume', 'tic','day']]
                    try:
                        bond_data = data_process(bond_data,True)
                    except Exception as e:
                        bond_data = data_process(bond_data,False)
                    all_bond_data.append(bond_data)
                    print(f"Data fetched for bond {bond_code}.")
                except Exception as e:
                    print(f"Error fetching data for bond {bond_code}: {e}")
                    continue
            # 合并所有可转债数据
            if all_bond_data:
                all_bonds_data_combined = pd.concat(all_bond_data)
                # 合并宏观数据
                data = self.robust_structure(all_bonds_data_combined)
                data['type'] = 2
                data.to_csv("./his_data/bond_data_from_ak.csv")
                return data
            else:
                print("No bonds data fetched.")
                return None
    
        except Exception as e:
            print(f"Error fetching convertible bond data: {e}")
            return None
        
    def get_stock_data_from_yf(self,start_date, end_date):
        try:
            data = YahooDownloader(ticker_list=self.ticker_list_yf, start_date=start_date, end_date=end_date).fetch_data()
            data['date'] = pd.to_datetime(data['date'])
            data['date'] = data['date'].dt.tz_localize('Asia/Shanghai')
            data['day'] = data['date'].dt.weekday
            try:
                data = data_process(data,True)
            except Exception as e:
                data = data_process(data,False)
            # 合并宏观数据
            data = self.robust_structure(data)
            data['type'] = 1
            # 合并宏观数据
            data.to_csv("./his_data/stock_data_from_yf.csv")
            return data
        except Exception as e:
            print(f"Error fetching stock_data_from_yf: {e}")
            return None
    
    def data_split(self,processed_full, start_date, end_date):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        trimmed_df = processed_full[(processed_full['date'] >= start_date) & (processed_full['date'] <= end_date)]
        return trimmed_df
    
    def upload_data(table_name,df):
        conn = hive.Connection(host="172.18.0.3", port=10000, username="hdfs")
        cursor = CONN.cursor()
        database_name = 'test_dataset'
            # Create table dynamically from DataFrame columns
        columns = ', '.join([f"`{col}` STRING" for col in df.columns])  # Assuming all columns are STRING, change types accordingly
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.{table_name} (
            {columns}
        ) STORED AS TEXTFILE
        LOCATION '/user/hive/warehouse/hive_data/finance_data/'
        """
        cursor.execute(create_table_query)
        
        # Insert data dynamically into the Hive table
        insert_query = f"""
        INSERT INTO TABLE {database_name}.{table_name} ({', '.join(df.columns)})
        VALUES ({', '.join(['%s' for _ in df.columns])})
        """
        with conn.cursor() as cursor:
            for row in df.itertuples(index=False, name=None):  # Skip index and use tuple row format
                cursor.execute(insert_query, row)
        print("Data uploaded successfully!")
        return None

if __name__ == '__main__':
    print("ok")
    


