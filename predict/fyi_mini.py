import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_stock_data(ticker="AAPL", period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval) #dowmload stock data
    return df[['Close']] # only get the close price column for said stock

def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler() # normlaize to range 0 -> 1
    scaled_data = scaler.fit_transform(data) # fit and tansform the data

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i]) # create window 
                                                # x and then also y 
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler # return sequences targets and scalaras

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), #L1
        LSTM(50), # L2
        Dense(1) # output
    ])
    model.compile(optimizer='adam', loss='mse') # compile with adam
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape), # L1
        GRU(50), #L2
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse') # complile with adam
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs # connect with residuals

    # feed forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res # add residuals again

def build_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, num_layers=2, dropout=0.1):
    inputs = layers.Input(shape=input_shape) # input layer
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x) # pooling over time steps
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs) # define
    model.compile(optimizer="adam", loss="mse") # complile with adam
    return model


def predict_next_day(model, recent_data, scaler): # predicts
    pred = model.predict(np.expand_dims(recent_data, axis=0))
    return scaler.inverse_transform(pred)[0][0] # rescales

def main():
    ticker = input("Enter stock: ").upper()
    print(f"Fetching data for {ticker}...")
    df = get_stock_data(ticker)

    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df.values) # normalize and prepare 

    #model = build_gru_model((X.shape[1], 1))
    #model = build_model((X.shape[1], 1))
    #model.fit(X, y, epochs=1000, batch_size=32, verbose=1)

    model = build_transformer_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32) # train

    next_price = predict_next_day(model, X[-1], scaler) # predict
    print(f"Predicted closing price for next day: ${next_price:.2f}")

    # plotting 
    plt.plot(df.index[-100:], df['Close'].values[-100:], label='Actual')
    plt.axhline(y=next_price, color='r', linestyle='--', label='Predicted Next Close')
    plt.title(f"{ticker} Price Prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
