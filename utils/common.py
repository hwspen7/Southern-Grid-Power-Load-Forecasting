import pandas as pd
import numpy as np

def data_preprocessing(path):
    """
    1. Load the dataset
    2. Convert the time column to the format: 2024-12-20 09:00:00
    3. Sort the data in ascending order by time
    4. Remove duplicate rows

    :param path: Path to the input CSV file
    :return: Preprocessed DataFrame
    """

    data = pd.read_csv(path)

    # Format the time column
    data['time'] = pd.to_datetime(data['time']).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Sort by time in ascending order
    data.sort_values(by=['time'], inplace=True)

    # Remove duplicates rows
    data.drop_duplicates(subset='time', inplace=True)

    return data

if __name__ == '__main__':
    data_preprocessing('../data/train.csv')

