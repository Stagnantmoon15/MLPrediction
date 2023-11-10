
# Stock Price Prediction Web Application

## Overview
This application is a Flask-based web service that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. It utilizes historical stock data fetched from Yahoo Finance (`yfinance` library) and employs `keras` for building and training the LSTM model.

## Features
- Fetch historical stock data for a specified symbol and date range.
- Predict future stock prices using LSTM neural network.
- Plot and compare historical and predicted stock price data.

## Installation

### Prerequisites
- Python 3.x
- Flask
- yfinance
- NumPy
- scikit-learn
- Keras

### Setup
To set up the application, you need to install the necessary Python packages. You can do this by running:

```
pip install flask yfinance numpy scikit-learn keras matplotlib
```

## Usage
To start the application, run the following command in the terminal:

```
python app.py
```

Navigate to `http://localhost:5000/` in your web browser. Enter a stock symbol and the application will display the historical and predicted stock prices.

## Application Structure

- `app.py`: The main file that contains Flask application and LSTM model implementation.
- `templates/`: Directory containing HTML templates for the web interface.
- `static/`: Directory for static files (if any).

## LSTM Model
The LSTM model is designed to predict stock prices based on historical data. It uses a sequence of the last 30 days' adjusted closing prices to predict the next day's price.

## Limitations
- Predictions are based solely on historical price data and do not consider other market factors.
- The model's accuracy is not guaranteed and should not be used for actual trading decisions.

## Additional Resources
- For more insights and understanding of stock price prediction using LSTM, check out this helpful video https://www.youtube.com/watch?v=R11_mg9R02c.
- What we tried to use for Deployment as a test see here: https://github.com/Stagnantmoon15/StockMLWebApp

