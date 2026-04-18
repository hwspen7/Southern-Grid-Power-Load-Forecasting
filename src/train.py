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

def analysis_data(ana_data):
    ana_data = ana_data.copy(deep=True)

    # Inspect the overall dataset
    print(ana_data.info())
    print(ana_data.head())

    # Overall distribution of power load
    fig = plt.figure(figsize=(20,40))
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'].values, bins=100, color='blue')
    ax1.set_title('Histogram of Power Load Distribution')
    ax1.set_xlabel('Power Load (MW)')
    ax1.set_ylabel('Frequency')

    # Average load trend by hour
    # Extract hour (HH) from the time string (format: YYYY-MM-DD HH:MM:SS)
    ana_data['hour'] = ana_data['time'].str[11:13]
    hour_load_mean = ana_data.groupby(['hour'])['power_load'].mean()
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean, color='blue')
    ax2.set_title('Average Power Load Trend by Hours')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Power Load (MW)')

    # Average load trend by month
    ax3 = fig.add_subplot(413)
    ana_data['month'] = ana_data['time'].str[5:7]
    data_month_avg = ana_data.groupby(['month'])['power_load'].mean()
    ax3.plot(data_month_avg, color='blue')
    ax3.set_title('Average Power Load Trend by Months')
    ax3.set_xlabel('Months')
    ax3.set_ylabel('Power Load (MW)')

    # Compare average load between weekdays and weekends
    ax4 = fig.add_subplot(414)
    """
    Extract weekday from the time column (Monday=0, ..., Sunday=6)
    Create a workday indicator: 1 for weekdays (Mon-Fri), 0 for weekends
    """
    ana_data['weekday'] = ana_data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    ana_data['workday']=ana_data['weekday'].apply(lambda x: 1 if x <= 4 else 0)

    power_load_workday_avg = ana_data[ana_data['weekday'] == 1].power_load.mean()
    power_load_weekendk_avg = ana_data[ana_data['weekday'] == 0].power_load.mean()

    ax4.bar(x=['Average Weekday Load', 'Average Weekend Load'],
            height=[power_load_workday_avg, power_load_weekendk_avg])
    ax4.set_title('Comparison of Average Weekday Load and Weekend Load')
    ax4.set_xlabel('Weekdays')
    ax4.set_ylabel('Average Load (MW)')

    plt.savefig('../data/figures/power_load_analysis.png')
    fig.show()


if __name__ == '__main__':
    power_load = PowerLoadModel(data_path=r'../data/train.csv')

    analysis_data(power_load.data_source)

