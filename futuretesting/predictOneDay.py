import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Function to fetch stock data for a given symbol
def get_stock_data(stock_symbol):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"Error: No data available for {stock_symbol}. Please enter a valid stock symbol.")
            return None
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


# Define the date range
start_date = "2018-01-01"
end_date = "2023-11-05"

# Ask the user to enter a stock symbol
while True:
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")
    df = get_stock_data(stock_symbol)
    if df is not None:
        break  # Valid symbol, exit the loop


# Create a MinMaxScaler to scale the data
scaler = MinMaxScaler()
df['Adj Close'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

# Define the sequence length (number of past days to consider)
sequence_length = 30

# Create sequences of data for training
data = []
target = []
for i in range(len(df) - sequence_length):
    data.append(df['Adj Close'].values[i:i + sequence_length])
    target.append(df['Adj Close'].values[i + sequence_length])

data = np.array(data)
target = np.array(target)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
train_target = target[:split_index]
test_data = data[split_index:]
test_target = target[split_index:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_data, train_target, batch_size=64, epochs=20)

# Predict stock prices for one day
predictions = model.predict(test_data)
predicted_price = predictions[-1][0]

# Inverse transform the predicted price to get the actual price
predicted_price = scaler.inverse_transform(np.array([[predicted_price]]))[0][0]

# Plot the actual vs. predicted stock prices for one day
test_target = scaler.inverse_transform(test_target.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(test_target, label='Actual')
plt.plot([len(test_target) - 1], [predicted_price], marker='o', markersize=5, label='Predicted', color='red')
plt.legend()
plt.title(f'Stock Price Prediction for {stock_symbol} (One Day)')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.show()

print(f"Predicted stock price for one day: {predicted_price}")


