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


def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            return None, "No data available for the provided symbol."
        return df, None
    except Exception as e:
        return None, str(e)


def predict_future_prices(model, last_sequence, days_to_predict, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_to_predict):
        current_sequence_reshaped = np.array(current_sequence).reshape(1, -1, 1)
        next_day_prediction = model.predict(current_sequence_reshaped)
        future_predictions.append(next_day_prediction[0][0])
        current_sequence.append(next_day_prediction[0][0])
        current_sequence.pop(0)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions


def predict_stock_price(stock_symbol, start_date, end_date, sequence_length):
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

    # Prepare for future price prediction
    last_sequence = list(df['Adj Close'].values[-sequence_length:])
    future_prices = predict_future_prices(model, last_sequence, 90, scaler)

    return future_prices, None


# Define the stock symbol and date range
start_date = "2018-01-01"
end_date = "2023-12-01"
sequence_length = 30


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        df, error_message = get_stock_data(stock_symbol, start_date, end_date)
        if df is not None:
            future_prices, error_message = predict_stock_price(stock_symbol, start_date, end_date, sequence_length)

            if future_prices is not None:
                day_30_prediction = future_prices[29][0]
                day_60_prediction = future_prices[59][0]
                day_90_prediction = future_prices[89][0]

                # Plot the historical and predicted stock price data
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['Adj Close'], label='Historical Data', color='blue')
                plt.title(f'Historical Stock Prices for {stock_symbol}')
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

                return render_template("index.html", plot=img_data, day_30=day_30_prediction, day_60=day_60_prediction,
                                       day_90=day_90_prediction)
            else:
                return render_template("index.html", error_message=error_message)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)