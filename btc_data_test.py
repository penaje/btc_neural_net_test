import re
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

pd.set_option("display.max_columns", None, 'display.max_rows', 25)
pd.set_option('display.width', 100)

df = pd.read_csv('cleaned_btc_data.csv')

# Convert the date strings into a date in Pandas
df['date'] = pd.to_datetime(df['date'], errors='raise')

print(df.head(15))
print(df.tail(15))


# plt.plot(df['close'])
# # Invert the x-axis
# plt.gca().invert_xaxis()
#
# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.title("BTC Closing Price 2017 - Present")
# plt.show()

# Oldest Data is Training Data
train_data = df.iloc[530:1881]

# Newer Data is Testing Data
test_data = df.iloc[0:530]

print("\nTEST\n")
print(test_data.head(15))
print(test_data.tail(15))

print("\nTRAIN\n")
print(train_data.head(15))
print(train_data.tail(15))

# model = keras.Sequential()
# model.add(keras.layers.LSTM(units=32, return_sequences=True, input_shape=(99, 1)))



