# Test

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2022-01-01"

df = yf.download(stock_symbol, start=start_date, end=end_date)


scaler = MinMaxScaler()
df['Adj Close'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

# Define the sequence length (number of past days to consider)
sequence_length = 30

# Create sequences of data for training
data = []
target = []
for i in range(len(df) - sequence_length):
    data.append(df['Adj Close'].values[i:i+sequence_length])
    target.append(df['Adj Close'].values[i+sequence_length])

data = np.array(data)
target = np.array(target)


split_ratio = 0.8
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
train_target = target[:split_index]
test_data = data[split_index:]
test_target = target[split_index:]


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_data, train_target, batch_size=64, epochs=20)


predictions = model.predict(test_data)
predictions = scaler.inverse_transform(predictions)

# Plot the actual vs. predicted stock prices
test_target = scaler.inverse_transform(test_target.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(test_target, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
