import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

pd.set_option("display.max_columns", None, 'display.max_rows', 25)
pd.set_option('display.width', 100)


def format_data(filename):
    """Input the csv file name in quotation marks, and it will format the data to be used in the model, returns a
    Pandas Data Frame"""
    df = pd.read_csv(filename)
    # print(df.head(15))

    # Convert the date strings into a date in Pandas
    # May need to adjust caps and/or implement to_lowercase or something
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')

    # Use the date as the index
    # This is needed for the Preprocessing Keras function
    df.set_axis(df['Date'], inplace=True)

    # Remove other columns except the date
    df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

    # print("\nData Cleaned\n")

    return df

def scale_data(old_array):
    """Takes a NumPy Array and returns the scaled array in proper format"""
    # Import the Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    new_array = scaler.fit_transform(old_array)
    new_array = new_array[~np.isnan(new_array)]
    new_array = new_array.reshape(-1, 1)

    return new_array


