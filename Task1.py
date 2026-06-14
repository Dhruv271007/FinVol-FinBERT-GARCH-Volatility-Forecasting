import yfinance as yf
import numpy as np
import pandas as pd
import os 

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA"]

if os.path.exists("price_data.csv"):
    data = pd.read_csv("price_data.csv", header=[0,1], index_col=0, parse_dates=True)
else:
    data = yf.download(tickers, start="2020-06-05", end="2026-06-05")
    data.to_csv("price_data.csv")

close = data["Close"]
log_returns = close.pct_change().apply(lambda x: np.log(1 + x))
realized_vol = log_returns.rolling(window=21).std() * np.sqrt(252)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(realized_vol.describe())