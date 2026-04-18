import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 15

class PowerLoadModel:
    def __init__(self, data_path):
        logfile_name = 'train' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logfile = Logger('../',logfile_name).get_logger()
        self.logfile.info('Start creating model...')
        self.data_source = data_preprocessing(data_path)

if __name__ == '__main__':
    power_load = PowerLoadModel(data_path=r'../data/train.csv')
    print(power_load.data_source.head())

