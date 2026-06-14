import pandas as pd
import numpy as np

tickers=["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","TSLA"]
for ticker in tickers :
    df = pd.read_csv(f"{ticker}_sentiment.csv")
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_sentiment = df.groupby('date')['sentiment'].mean()
    daily_sentiment = daily_sentiment.shift(1)  # lag by how many days?
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment.columns = ['date', 'sentiment_lag1']
    daily_sentiment.to_csv(f"{ticker}_daily_sentiment.csv", index=False)
    print(f"{ticker}: {len(daily_sentiment)} days of sentiment saved")

