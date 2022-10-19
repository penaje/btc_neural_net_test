import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
import data_cleaner
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None, 'display.max_rows', 30)
pd.set_option('display.width', 150)

# Call out data clean function to return our data frame
df = pd.read_csv('btc_data_2014_2022.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='raise')
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Adj Close'], inplace=True)

print("\nDATAFRAME\n")
print(df.head(2))


def scale_data(old_array):
    """Takes a NumPy Array and returns the scaled array in proper format"""
    # Import the Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    new_array = scaler.fit_transform(old_array)
    new_array = new_array[~np.isnan(new_array)]
    new_array = new_array.reshape(-1, 1)

    return new_array


# Create the NumPy Arrays
open_data = df.Open.values.reshape(-1, 1)
high_data = df.High.values.reshape(-1, 1)
low_data = df.Low.values.reshape(-1, 1)
volume_data = df.Volume.values.reshape(-1, 1)
close_data = df.Close.values.reshape(-1, 1)

# Scale the NumPy Arrays
open_data = scale_data(open_data)
high_data = scale_data(high_data)
low_data = scale_data(low_data)
volume_data = scale_data(volume_data)
close_data = scale_data(close_data)

# Combine the data
combined_data = np.hstack((volume_data, open_data, high_data, low_data, close_data))

# Create the target data from closing data
target_data = df.Close.values.reshape(-1, 1)

# Calculate Split Percent
split_percent = 0.80
split = int(split_percent * len(combined_data))

# Split the closing price Numpy array
close_train = combined_data[:split]
close_test = combined_data[split:]

# Split the target Data
target_train = close_data[:split]
target_test = close_data[split:]


# Split the date data
date_train = df.Date[:split]
date_test = df.Date[split:]

# Look back period is 100 days
look_back = 14

train_generator = TimeseriesGenerator(close_train, target_train, length=look_back, batch_size=32)
