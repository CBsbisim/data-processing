import os
import pandas as pd
import numpy as np
import akshare as ak
import ray
from models.data_processing import DP
from models.env_trading import FinanceTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib import font_manager
# 设置字体为已安装的中文字体
# 设置字体路径
font_path = './models/msyh.ttc'  # 替换成你的字体路径
font_prop = font_manager.FontProperties(fname=font_path)

check_and_make_directories([TRAINED_MODEL_DIR])

os.environ['RAY_memory_usage_threshold'] = '0.9'

# 去除不相关因子
MACRO_INDICATORS = [ 'consumer_confidence_usa','consumer_confidence_cn', 'cpi_cn','news_sentiment_scope']

def base():
    train = pd.read_csv('./his_data/train_data.csv')
    train = train.set_index('0')
    train.index.names = ['']
    finance_dimension = len(train.tic.unique())
    state_space = 1 + 2 * finance_dimension + len(INDICATORS) * finance_dimension +len(MACRO_INDICATORS)+1*finance_dimension
    print(f"Finance Dimension: {finance_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * finance_dimension
    num_finance_shares = [0] * finance_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_finance_shares": num_finance_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_weight": 1.2, # 股票的权重
        "bond_weight": 0.8,  # 债券的权重
        "finance_dim": finance_dimension,
        "tech_indicator_list": INDICATORS,
        "macro_indicator_list":MACRO_INDICATORS,
        "action_space": finance_dimension,
        
        "reward_scaling": 1e-4
    }
    e_train_gym = FinanceTradingEnv(df = train, **env_kwargs)
    obs = e_train_gym.reset()
    env_train, _ = e_train_gym.get_sb_env()
    return env_train

ENV_TRAIN = base()

def agent_a2c():
    agent = DRLAgent(env = ENV_TRAIN)
    model_a2c = agent.get_model("a2c")
    # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)
    trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=30000)
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")


def agent_ddpg():
    agent = DRLAgent(env = ENV_TRAIN)
    model_ddpg = agent.get_model("ddpg")
    # set up logger
    tmp_path = RESULTS_DIR + '/ddpg'
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)
    trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=30000)
    trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")


def agent_td3():
    agent = DRLAgent(env = ENV_TRAIN)
    TD3_PARAMS = {"batch_size": 100, 
                  "buffer_size": 1000000, 
                  "learning_rate": 0.01}
    
    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + '/td3'
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)
    trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=30000)
    trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")


def agent_sac():
    agent = DRLAgent(env = ENV_TRAIN)
    SAC_PARAMS = {
        "batch_size": 256,
        "buffer_size": 1000000,
        "learning_rate": 0.05,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
    
    model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + '/sac'
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)
    trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=50000)
    trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")


@ray.remote
def model_pretraining(func, flag):
    try:
        print(f"Pretraining model from {func.__name__}...") if flag else None
        func() if flag else None
        print(f"Completed model {func.__name__}.") if flag else None
        return None
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return None

def model_save(dict_):
    # Set the corresponding values to 'True' for the algorithms that you want to use
    if_using_a2c = dict_['if_using_a2c']
    if_using_ddpg = dict_['if_using_ddpg']
    if_using_td3 = dict_['if_using_td3']
    if_using_sac = dict_['if_using_sac']

    tasks = [
        (agent_a2c, if_using_a2c),
        (agent_ddpg, if_using_ddpg),
        (agent_sac, if_using_sac),
        (agent_td3, if_using_td3),
    ]
    for func, flag in tasks:
        future = model_pretraining.remote(func, flag)
        # Collect all results in parallel
        try:
            # Optional: Reinitialize Ray if you need it for further tasks
            print("Reinitializing Ray...")
            ray.init(num_cpus=32, num_gpus=0, ignore_reinit_error=True)
            ray.get(future)
            print("Shutting down Ray to refresh memory...")
            ray.shutdown()
            print(f"Model {func.__name__} pretraining completed.")
        except Exception as e:
            print(f"Error during task execution: {e}")
            data_results = []


if __name__ == "__main__":
    print("ok")
    