from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

app = Flask(__name__)

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
        return None

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

    # Predict stock prices for the specified number of days
    predictions = model.predict(test_data)
    predicted_price = predictions[-1][0]

    # Inverse transform the predicted price to get the actual price
    predicted_price = scaler.inverse_transform(np.array([[predicted_price]]))[0][0]

    return predicted_price


@app.route('/', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        days_to_predict = [int(day) for day in request.form.getlist('days_to_predict')]
        start_date = "2018-01-01"
        end_date = "2023-11-05"
        sequence_length = 30
        predictions = {}
        df = get_stock_data(stock_symbol, start_date, end_date)
        if df is not None:
            for days in days_to_predict:
                predicted_price = predict_stock_price(stock_symbol, start_date, end_date, sequence_length, days)
                predictions[days] = predicted_price
        return render_template('index.html', predictions=predictions, stock_symbol=stock_symbol)

    return render_template('index.html', predictions=None, stock_symbol=None)


if __name__ == '__main__':
    app.run(debug=True)
