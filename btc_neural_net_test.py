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

# Code adapted from: "https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks
# -74674e289816"
# and
# "https://python.plainenglish.io/super-simple-neural-network-for-bitcoin-price-prediction-in-python-8c2cd46d11a7"

# Call out data clean function to return our data frame
df = data_cleaner.format_data('btc_data_2014_2022.csv')

print("\nDATAFRAME\n")
print(df.head(15))
print(df.tail(15))

# Convert DataFrame in Numpy Array and reshape data for use in the model
close_data = df.Close.values.reshape((-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))

close_data = scaler.fit_transform(close_data)
close_data = close_data[~np.isnan(close_data)]
close_data = close_data.reshape(-1, 1)

print("\nNUMPY ARRAY\n")
print(close_data)

# Calculate Split Percent
split_percent = 0.80
split = int(split_percent * len(close_data))

# Split the closing price Numpy array
close_train = close_data[:split]
close_test = close_data[split:]

print('\nCLOSE TRAIN\n')
print(close_train)

# Split the date data
date_train = df.Date[:split]
date_test = df.Date[split:]

# Look back period is 100 days
look_back = 14

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=32)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

# # Test Model #1
# model = Sequential()
# model.add(LSTM(units=32, return_sequences=True, input_shape=(look_back, 1), dropout=0.2))
# model.add(LSTM(units=32, return_sequences=True, dropout=0.2))
# model.add(LSTM(units=32, dropout=0.2))
# model.add(Dense(units=1))

# Test Model #2
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(look_back, 1), dropout=.2))
model.add(Dense(1))

# Test Model #3
# model = Sequential()
# model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(look_back, 1)))
# model.add(LSTM(units=60, activation='relu', return_sequences=True, dropout=0.2))
# model.add(LSTM(units=80, activation='relu', return_sequences=True, dropout=0.3))
# model.add(LSTM(units=120, activation='relu', dropout=0.4))
# model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse')

num_epochs = 20

model.fit(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

# Make the Final Graph
trace1 = go.Scatter(x=date_train, y=close_train, mode='lines', name='Data')
trace2 = go.Scatter(x=date_test, y=prediction, mode='lines', name='Predicted Price')
trace3 = go.Scatter(x=date_test, y=close_test, mode='lines', name='Actual Price')
layout = go.Layout(title="BTC/USDT 2014 - 2022", xaxis={'title': "Date"}, yaxis={'title': "Close Price"})
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()
