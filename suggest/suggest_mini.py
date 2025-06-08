import time
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from datetime import datetime

# the config form
symbol = "AAPL"
starting_balance = 1000.0 # how much you wanna do
position = 0  # how many shares we hold
cash = starting_balance
rsi_buy = 30
rsi_sell = 70

def get_live_data(ticker):
    df = yf.download(ticker, period="2d", interval="1m", progress=False)
    print(f"Data fetched: {df.shape} rows, columns: {df.columns.tolist()}") # debug
    print(f"Close column type: {type(df['Close'])}, shape: {df['Close'].shape}")
    return df

def compute_rsi(df):
    close_series = df['Close'].squeeze()
    rsi = RSIIndicator(close_series, window=14).rsi()
    return rsi.iloc[-1]

def simulate_trade(action, price):
    global position, cash
    if action == "BUY" and cash >= price:
        position += 1
        cash -= price
        print(f"BUY --- 1 share at ${price:.2f}") # notify
    elif action == "SELL" and position > 0:
        position -= 1
        cash += price
        print(f"SELL --- 1 share at ${price:.2f}") # notify
    else:
        print("holding") # hold
 
def report(price):
    total = cash + position * price
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Price: ${price:.2f} | Shares: {position} | Cash: ${cash:.2f} | Total In: ${total:.2f}")

def main():
    while True:
        try:
            df = get_live_data(symbol)
            current_price = df['Close'].iloc[-1].item()  # get scalar float safely
            rsi = compute_rsi(df)
            print(f"RSI: {rsi:.2f}")

            if rsi < rsi_buy:
                simulate_trade("BUY", current_price)
            elif rsi > rsi_sell:
                simulate_trade("SELL", current_price)
            else:
                print("HOLD")

            report(current_price)
            time.sleep(20)  

        except Exception as e:
            print("error:", e)
            time.sleep(20) # wait

if __name__ == "__main__":
    main()
