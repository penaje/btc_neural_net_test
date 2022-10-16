import pandas as pd

pd.set_option("display.max_columns", None, 'display.max_rows', 25)
pd.set_option('display.width', 100)


def format_data(filename):
    """Input the csv file name and it will format the data to be used in the model, returns the Data Frame"""
    df = pd.read_csv(filename)
    print(df.head(15))

    # Convert the date strings into a date in Pandas
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')

    # Use the date as the index
    df.set_axis(df['Date'], inplace=True)

    # Remove other columns except the date
    df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

    print("\nData Cleaned\n")

    return df


format_data('btc_data_2014_2022.csv')

