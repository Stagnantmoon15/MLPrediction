import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Function to fetch stock data for a given symbol
def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"Error: No data available for {stock_symbol}. Please enter a valid stock symbol.")
            return None
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to predict stock price for a given number of days
def predict_stock_price(stock_symbol, start_date, end_date, sequence_length, days):
    df = get_stock_data(stock_symbol, start_date, end_date)
    if df is None:
        return

    # Create a MinMaxScaler to scale the data
    scaler = MinMaxScaler()
    df['Adj Close'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

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

    # Predict stock prices for the available historical data
    predictions = model.predict(data)  # Predict using historical data
    predicted_prices = scaler.inverse_transform(predictions)

    return predicted_prices


# Define the stock symbol and date range
start_date = "2018-01-01"
end_date = "2023-11-05"
sequence_length = 30

while True:
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")
    df = get_stock_data(stock_symbol, start_date, end_date)
    if df is not None:
        break  # Valid symbol, exit the loop

# Predict stock prices for the available historical data
predicted_prices = predict_stock_price(stock_symbol, start_date, end_date, sequence_length, 0)


# Plot the historical and predicted stock price data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Adj Close'], label='Historical Data', color='blue')
plt.plot(df.index[sequence_length:], predicted_prices, label='Predicted Data', color='red', linestyle='--')
plt.title(f'Historical vs Predicted Stock Prices for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)
plt.show()
