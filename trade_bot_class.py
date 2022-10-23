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
from binance import Client
from binance.enums import *
from time import sleep
import json
import datetime

pd.set_option("display.max_columns", None, 'display.max_rows', 30)
pd.set_option('display.width', 150)

api_public = 'sxV243LlOdPF8hzKGfiZWjnN4PZAfwkQRpNDnYlqHYHmWh8HGqRQlMCxzLjYMxb7'
api_secret = 'B9duqs0DkZE2wQqohygM7IxHleZsgZ3ov7o6ObmWMZz8me8FBFfHg7Ys4Sdagnvs'


# client = Client(api_key, api_secret, testnet=True, tld='us')

class TradeBot:
    """Trading Bot Class"""

    def __init__(self):
        self._btc_balance = None
        self._last_btc_price = None
        self._api_key = None
        self._api_secret = None
        self._client = None
        self._model = None
        self._price_data = None

    def set_keys(self, public_key, secret_key):
        """Sets the keys for the account linked to the bot"""
        self._api_key = public_key
        self._api_secret = secret_key

    def create_client(self):
        """Create a client to be used with the bot, returns the client object"""
        self._client = Client(self._api_key, self._api_secret, testnet=True, tld='us')

    def get_client(self):
        """Returns the Client object"""
        return self._client

    def update_btc_balance(self):
        """Updates the value in the BTC balance"""
        self._btc_balance = self._client.get_asset_balance(asset='BTC')['free']

    def get_btc_balance(self):
        """Updates the balance and returns the free BTC balance"""
        self.update_btc_balance()
        return self._btc_balance

    def get_current_btc_price(self):
        """Updates the latest btc price and prints it out"""
        self._last_btc_price = self._client.get_symbol_ticker(symbol='BTCUSDT')['price']
        print(self._last_btc_price)

    def get_balance_of(self, asset_name):
        """Returns the balance of the specified asset"""
        asset = asset_name + "USDT"
        return self._client.get_asset_balance(asset=asset_name)['free']

    def get_price_of(self, asset_name):
        """Returns the latest price of the requested asset"""
        asset = asset_name + "USDT"
        return self._client.get_symbol_ticker(symbol=asset)['price']

    def get_btc_data_csv(self, num_days):
        """Get the designated number of days past data, return a data frame"""
        data = pd.read_csv(
            'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1635010346&period2=1666546346'
            '&interval=1d&events=history&includeAdjustedClose=true')
        df = data.tail(num_days)
        df['Date'] = pd.to_datetime(df['Date'], errors='raise')
        df.set_axis(df['Date'], inplace=True)
        df.drop(columns=['Adj Close', 'Date'], inplace=True)
        self._price_data = df

    def add_prediction_model(self, fileName):
        """Returns the prediction model"""
        # Load in our model
        self._model = load_model(fileName)

    def get_tomorrows_price(self):
        """Returns tomorrows price after loading the model and dataframe"""

        look_back = 45

        # Create the NumPy Arrays
        open_data = self._price_data.Open.values.reshape(-1, 1)
        high_data = self._price_data.High.values.reshape(-1, 1)
        low_data = self._price_data.Low.values.reshape(-1, 1)
        volume_data = self._price_data.Volume.values.reshape(-1, 1)
        close_data = self._price_data.Close.values.reshape(-1, 1)

        # Scale the NumPy Arrays & Graph Data
        open_data = helper_functions.scale_data(open_data)
        high_data = helper_functions.scale_data(high_data)
        low_data = helper_functions.scale_data(low_data)
        volume_data = helper_functions.scale_data(volume_data)

        # This is used for the combined data and as the Target Data
        close_data = helper_functions.scale_data(close_data)

        # Combine the data
        combined_data = np.hstack((volume_data, open_data, high_data, low_data, close_data))

        last_period_data = combined_data.reshape((1, look_back, 5))

        new_prediction = self._model.predict(last_period_data[0][None])

        new_prediction = helper_functions.unscale_data(new_prediction).reshape((-1))

        predication_as_int = new_prediction.astype(float)

        print("Tomorrow Price Prediction: ", predication_as_int[0])


bot1 = TradeBot()
bot1.set_keys(api_public, api_secret)
bot1.get_btc_data_csv(45)
bot1.add_prediction_model('test_model.h5')
bot1.get_tomorrows_price()
