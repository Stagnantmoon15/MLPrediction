import os
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from flask import Flask, request, render_template
from matplotlib.figure import Figure
from io import BytesIO
import base64

app = Flask(__name__)


# Function to fetch stock data
def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            return None, "No data available for the provided symbol."
        return df, None
    except Exception as e:
        return None, str(e)


# Function to predict stock price
def predict_stock_price(stock_symbol, start_date, end_date, sequence_length):
    df, error_message = get_stock_data(stock_symbol, start_date, end_date)
    if df is None:
        return None, error_message

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

    data = []
    for i in range(sequence_length, len(scaled_data)):
        data.append(scaled_data[i-sequence_length:i, 0])

    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    model_path = f'models/{stock_symbol}.h5'
    if not os.path.exists(model_path):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(data, scaled_data[sequence_length:], epochs=20, batch_size=32)

        if not os.path.exists('../models'):
            os.makedirs('../models')
        model.save(model_path)
    else:
        model = load_model(model_path)

    # Predicting all the data to plot historical and predicted data
    predictions = model.predict(data)
    predicted_prices = scaler.inverse_transform(predictions)

    return df['Adj Close'], predicted_prices.flatten(), None


# Function to forecast future prices
def forecast_future_prices(stock_symbol, sequence_length, days_to_forecast):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=sequence_length + days_to_forecast)).strftime('%Y-%m-%d')
    df, _ = get_stock_data(stock_symbol, start_date, end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

    model_path = f'models/{stock_symbol}.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        return None, "Model not found"

    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

    forecasted_prices = []
    for _ in range(days_to_forecast):
        predicted_price = model.predict(last_sequence)
        forecasted_prices.append(predicted_price[0,0])
        # Update the last_sequence
        last_sequence = np.append(last_sequence[:, 1:, :], [[predicted_price]], axis=1)
        last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

    forecast = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))

    return forecast.flatten()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        start_date = "2018-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        sequence_length = 60

        historical_prices, predicted_prices, error_message = predict_stock_price(stock_symbol, start_date, end_date, sequence_length)

        if error_message is None:
            # Generate plot
            fig = Figure()
            ax = fig.subplots()
            ax.plot(historical_prices.index, historical_prices, label='Historical Data', color='blue')
            ax.plot(historical_prices.index[sequence_length:], predicted_prices, label='Predicted Data', color='red', linestyle='--')
            ax.set_title(f'Historical vs Predicted Stock Prices for {stock_symbol}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Adjusted Close Price')
            ax.legend()

            # Convert plot to PNG image
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            return render_template("index.html", plot=img_data)
        else:
            return render_template("index.html", error_message=error_message)

    return render_template("index.html")


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        sequence_length = 60
        days_to_forecast = 30

        forecasted_prices = forecast_future_prices(stock_symbol, sequence_length, days_to_forecast)

        return render_template("forecast.html", forecast=forecasted_prices)

    return render_template("forecast.html")


if __name__ == "__main__":
    app.run(debug=True)
