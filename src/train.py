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

def feature_engineering(ana_data):
    """
    Perform feature engineering on the input dataset and extract key features

    1. Extract time-related features: month and hour
    2. Extract recent load features within a given window size
    3. Extract the load value from the same time yesterday
    4. Remove samples with missing values
    5. Organize and return the feature columns

    :param ana_data: Input dataset
    """
    feature_data = ana_data.copy(deep=True)
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]

    '''
    Use one-hot encoding to convert categorical time features (hour and month) into binary variables
    so that the model can learn each category separately without assuming ordinal relationships
    '''
    hour_encoding = pd.get_dummies(feature_data['hour'])
    hour_encoding.columns = ['hour_' + str(i) for i in hour_encoding.columns]

    month_encoding = pd.get_dummies(feature_data['month'])
    month_encoding.columns = ['month_' + str(i) for i in month_encoding.columns]

    # Concatenate encoded features
    feature_data = pd.concat([feature_data, hour_encoding, month_encoding], axis=1)

    # Extract recent load values within a fixed window

    # load_1h_data = feature_data['power_load'].shift(1)
    # load_2h_data = feature_data['power_load'].shift(2)
    # load_3h_data = feature_data['power_load'].shift(3)
    # load_shift_data = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis=1)
    # load_shift_data.columns = ['load_1h', 'load_2h', 'load_3h']
    # feature_data = pd.concat([feature_data, load_shift_data], axis=1)

    window_size = 3
    shift_list = [feature_data['power_load'].shift(i) for i in range(1, window_size+1)]
    shift_data = pd.concat(shift_list, axis=1)
    shift_data.columns = ['previous_'+str(i) for i in range(1, window_size+1)]
    feature_data = pd.concat([feature_data, shift_data], axis=1)

    # Compute the same timestamp from yesterday
    feature_data['yesterday_time'] = feature_data['time'].apply(
        lambda x:(pd.to_datetime(x) - pd.to_timedelta('1D')).strftime('%Y-%d %H:%M:%S')
    )
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))
    feature_data.dropna(axis=0,inplace=True)

    feature_columns = list(hour_encoding.columns.append(month_encoding.columns)) +list(shift_data.columns) + ['yesterday_load']
    print(feature_columns)



if __name__ == '__main__':
    power_load = PowerLoadModel(data_path=r'../data/train.csv')

    analysis_data(power_load.data_source)
    feature_engineering(power_load.data_source)

