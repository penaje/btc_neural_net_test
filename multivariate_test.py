import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import helper_functions

# Set the Pandas display preferences
pd.set_option("display.max_columns", None, 'display.max_rows', 30)
pd.set_option('display.width', 150)

# Format data how we need it for multivariate
df = pd.read_csv('btc_data_2014_2022.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='raise')
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Adj Close'], inplace=True)

# Create the NumPy Arrays
open_data = df.Open.values.reshape(-1, 1)
high_data = df.High.values.reshape(-1, 1)
low_data = df.Low.values.reshape(-1, 1)
volume_data = df.Volume.values.reshape(-1, 1)
close_data = df.Close.values.reshape(-1, 1)

# Scale the NumPy Arrays & Graph Data
open_data = helper_functions.scale_data(open_data)
high_data = helper_functions.scale_data(high_data)
low_data = helper_functions.scale_data(low_data)
volume_data = helper_functions.scale_data(volume_data)

# This is used for the combined data and as the Target Data
close_data = helper_functions.scale_data(close_data)

# Combine the data
combined_data = np.hstack((volume_data, open_data, high_data, low_data, close_data))

## ***THIS PRE CLIPPING CAN BE REMOVED IF I GET A DIFFERENT .CSV FILE***
# Remove the front 25%, price doesn't move much before 2017
pre_split_percent = 0.25
pre_split = int(pre_split_percent * len(combined_data))

# Clip the combined data and target data
combined_data = combined_data[pre_split:]
close_data = close_data[pre_split:]

# Calculate Split Percent for the Training and Test partitioning
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
look_back = 30

# Prints the first 10 prices, for testing
# for x in range(10):
#     print("CLOSE TRAIN:   ", target_train[x])

# Create the training generator
train_generator = TimeseriesGenerator(close_train, target_train, length=look_back, batch_size=1)

# Prints the first 3 sets of data in the generator
# for i in range(3):
#     x, y = train_generator[i]
#     print('%s => %s' % (x, y))

# Create the testing data generator
test_generator = TimeseriesGenerator(close_test, target_test, length=look_back, batch_size=1)

# # Test Model #1
# model = Sequential()
# model.add(LSTM(units=32, return_sequences=True, input_shape=(look_back, 5), dropout=0.2))
# model.add(LSTM(units=32, return_sequences=True, dropout=0.2))
# model.add(LSTM(units=32, dropout=0.2))
# model.add(Dense(units=1))

# # Test Model #2
# model = Sequential()
# model.add(LSTM(units=32, activation='relu', input_shape=(1, 5), dropout=.2))
# model.add(Dense(1))


# Test Model #3
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(look_back, 5)))
model.add(LSTM(units=60, activation='relu', return_sequences=True, dropout=0.2))
model.add(LSTM(units=80, activation='relu', return_sequences=True, dropout=0.3))
model.add(LSTM(units=120, activation='relu', dropout=0.4))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse')

num_epochs = 10

model.fit(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict(test_generator)

close_train = target_train.reshape((-1))
close_test = target_test.reshape((-1))
prediction = prediction.reshape((-1))

# Make the Final Graph
trace1 = go.Scatter(x=date_train, y=close_train, mode='lines', name='Data')
trace2 = go.Scatter(x=date_test, y=prediction, mode='lines', name='Predicted Price')
trace3 = go.Scatter(x=date_test, y=close_test, mode='lines', name='Actual Price')
layout = go.Layout(title="BTC/USDT 2014 - 2022", xaxis={'title': "Date"}, yaxis={'title': "Close Price"})
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()
