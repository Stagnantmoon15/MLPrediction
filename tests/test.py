import matplotlib
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from io import BytesIO
import base64


matplotlib.use('Agg')
app = Flask(__name__)


# Function to fetch stock data for a given symbol
def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            return None, "No data available for the provided symbol."
        return df, None
    except Exception as e:
        return None, str(e)


# Function to predict stock price for a given number of days
def predict_stock_price(stock_symbol, start_date, end_date, sequence_length, days):
    df, error_message = get_stock_data(stock_symbol, start_date, end_date)
    if df is None:
        return None, error_message

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

    return predicted_prices, None


# Define the stock symbol and date range
start_date = "2018-01-01"
end_date = "2023-12-01"
sequence_length = 30


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        df, error_message = get_stock_data(stock_symbol, start_date, end_date)  # Fetch stock data
        if df is not None:
            predicted_prices, error_message = predict_stock_price(stock_symbol, start_date, end_date, sequence_length, 0)

            if predicted_prices is not None:
                # Plot the historical and predicted stock price data
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['Adj Close'], label='Historical Data', color='blue')
                plt.plot(df.index[sequence_length:], predicted_prices, label='Predicted Data', color='red', linestyle='--')
                plt.title(f'Historical vs Predicted Stock Prices for {stock_symbol}')
                plt.xlabel('Date')
                plt.ylabel('Adjusted Close Price')
                plt.legend()
                plt.grid(True)

                # Save the plot to a BytesIO object
                img_buf = BytesIO()
                plt.savefig(img_buf, format="png")
                img_buf.seek(0)
                img_data = base64.b64encode(img_buf.read()).decode()
                plt.close()

                return render_template("index.html", plot=img_data)

            else:
                return render_template("index.html", error_message=error_message)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)


