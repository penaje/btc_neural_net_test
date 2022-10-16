import re
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None, 'display.max_rows', 25)
pd.set_option('display.width', 100)

df = pd.read_csv('cleaned_btc_data.csv')

# Convert the date strings into a date in Pandas
df['date'] = pd.to_datetime(df['date'], errors='raise')

print(df.head(15))
print(df.tail(15))

plt.plot(df['close'])
# Invert the x-axis
plt.gca().invert_xaxis()

plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("BTC Closing Price 2017 - Present")
plt.show()