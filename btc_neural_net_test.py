import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Code adapted from: "https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816"

pd.set_option("display.max_columns", None, 'display.max_rows', 25)
pd.set_option('display.width', 100)

df = pd.read_csv('cleaned_btc_data.csv')

# Convert the date strings into a date in Pandas
df['date'] = pd.to_datetime(df['date'], errors='raise')

print(df.head(15))
print(df.tail(15))

# Reverse the data
df = df.iloc[::-1]

# Use the date as the index
df.set_axis(df['date'], inplace=True)
# Remove other columns
df.drop(columns=['open', 'high', 'low'], inplace=True)

print(df.head(15))
print(df.tail(15))


# Make Numpy Array and reshape data
close_data = df.close.values.reshape((-1, 1))


# Calculate Split Percent
split_percent = 0.80
split = int(split_percent * len(close_data))

# Split the closing price Numpy array
close_train = close_data[:split]
close_test = close_data[split:]

# Split the date Numpy Array
date_train = df.date[:split]
date_test = df.date[split:]

print("\nTESTING\n")
print(len(close_train))
print(len(close_test))

# Look back period is 30 days
look_back = 30

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(look_back, 1), dropout=.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

num_epochs = 50

model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x=date_train,
    y=close_train,
    mode='lines',
    name='Data'
)
trace2 = go.Scatter(
    x=date_test,
    y=prediction,
    mode='lines',
    name='Prediction'
)
trace3 = go.Scatter(
    x=date_test,
    y=close_test,
    mode='lines',
    name='Ground Truth'
)
layout = go.Layout(
    title="Google Stock",
    xaxis={'title': "Date"},
    yaxis={'title': "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()
