import os
import pandas as pd
import numpy as np
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


class PowerLoadPredictor(object):
    def __init__(self, file_path, log_name):
        logfile_name = 'predict' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
        self.logger = Logger('../', logfile_name).get_logger()
        self.data_source = data_preprocessing(file_path)
        self.time_load_dict = self.data_source.set_index('time')['power_load'].to_dict()

        self.logger.info(
            f"========== Extracting features for prediction time: {datetime.datetime.now().strftime('%Y%m%d%H%M%S')} ==========")


def pred_feature_extraction(data_dict, time, logger):
    """
    Extract prediction features while keeping the same feature column names
    used during model training

    :param data_dict: historical data in dictionary format, key = timestamp, value = load
    :param time: prediction timestamp as a string, format example: 2024-12-20 09:00:00
    :param logger: logger object
    :return: feature DataFrame and feature name list
    """
    # List of feature column names
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     'previous_1', 'previous_2', 'previous_3', 'yesterday_load']

    # Store hour features using a list
    hour_list = []
    pred_hour = time[11:13]
    for i in range(24):
        if pred_hour == feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)

    # Store month features using a list
    month_list = []
    pred_month = time[5:7]
    for i in range(24,36):
        if pred_month == feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)

    last_1h_time = (pd.to_datetime(time) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_time = (pd.to_datetime(time) - pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_time = (pd.to_datetime(time) - pd.Timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    last_day_time = (pd.to_datetime(time) - pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

    last_1h_load = data_dict.get(last_1h_time, 600)
    last_2h_load = data_dict.get(last_2h_time, 600)
    last_3h_load = data_dict.get(last_3h_time, 600)
    last_day_load = data_dict.get(last_day_time, 600)

    his_list = [last_1h_load, last_2h_load, last_3h_load, last_day_load]
    feature_list = [hour_list + month_list + his_list]

    feature_data = pd.DataFrame(feature_list, columns=feature_names)
    logger.info(f"Features extracted for {time}")
    return feature_data, feature_names

def prediction_plot(data):
    """
    Plot the predicted load curve and the true load curve over time to visualize prediction performance
    :param data:
    :return:
    """
    fig = plt.figure(figsize=(30,60))
    ax = fig.add_subplot()

    # Plot the real load over time
    ax.plot(data['Time'], data['Real Value'], label='Real Value')

    # Plot the predicted load over time
    ax.plot(data['Time'], data['Predicted Value'], label='Predicted Value')

    ax.set_xlabel('Time')
    ax.set_ylabel('Load')
    ax.set_title('Line Chart of Predicted Load and Real Load')
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    ax.legend()

    plt.xticks(rotation=45)
    plt.savefig('../data/figures/predicted_load_and_real_load.png')



if __name__ == '__main__':
    power_load_predictor = PowerLoadPredictor('../data/test.csv', 'predict')
    pred_obj = power_load_predictor.data_source

    # Load the trained model
    model = joblib.load('../model/xgb.pkl')

    evaluate_list = []
    pred_times_list = power_load_predictor.data_source[
        power_load_predictor.data_source['time'] >= '2015-08-01 00:00:00'
        ]['time']

    for pred_time in pred_times_list:
        data_his_dict = {
            k: v for k, v in power_load_predictor.time_load_dict.items() if k < pred_time
        }
        processed_data, feature_cols = pred_feature_extraction(
            data_his_dict,
            pred_time,
            power_load_predictor.logger
        )

        pred_value = model.predict(processed_data[feature_cols])[0]
        real_value = power_load_predictor.time_load_dict.get(pred_time)
        power_load_predictor.logger.info(
            f"Time: {pred_time}, True load: {real_value}, Predicted load: {pred_value}"
        )
        evaluate_list.append([pred_time, real_value, pred_value])

    evaluate_df = pd.DataFrame(
        evaluate_list,
        columns=['Time', 'Real Value', 'Predicted Value']
    )

    if evaluate_df.empty:
        print("evaluate_df is empty")
    else:
        mae_score = mean_absolute_error(evaluate_df['Real Value'], evaluate_df['Predicted Value'])
        print(f'MAE score: {mae_score}')
        power_load_predictor.logger.info(f'MAE score: {mae_score}')
        prediction_plot(evaluate_df)

