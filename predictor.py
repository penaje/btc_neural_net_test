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
from keras.models import load_model

# This is our look back period the model was trained on
look_back = 45

# Format data how we need it for multivariate
# We need to pull this from Binance in real time eventually...
df = pd.read_csv('btc_data_2014_2022.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='raise')
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Adj Close'], inplace=True)

# Take the last 45 days of DATA
# If using the generator for predictions add 1 to this
df = df.tail(look_back)

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

# Load in our model
model = load_model('test_model.h5')

last_period_data = combined_data.reshape((1, look_back, 5))

new_prediction = model.predict(last_period_data[0][None])

new_prediction = helper_functions.unscale_data(new_prediction).reshape((-1))

predication_as_int = new_prediction.astype(float)

print("New Prediction:", predication_as_int[0])

