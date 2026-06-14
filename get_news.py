from dotenv import load_dotenv
import os
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from datetime import datetime
import pandas as pd
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)


symbols = ["AAPL", "MSFT", "NVDA",  "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA"]

for ticker in symbols:
    all_news = []
    end_date = datetime(2026,6,5)
    if os.path.exists(f"{ticker}_news.csv"):
        df = pd.read_csv(f"{ticker}_news.csv")
    else:
        while True:
            request = NewsRequest(
                symbols=f"{ticker}",
                start=datetime(2020,6,5),
                end=end_date,
                limit=50,
            )
            news = client.get_news(request)
            articles = news.data['news']
    
            all_news.extend(articles)
            if len(articles) < 50:
                break
            end_date = articles[- 1].created_at
        print(len(all_news))
        df = pd.DataFrame([{
            'date': article.created_at,
            'headline': article.headline,
            'symbols': article.symbols
        } for article in all_news])
        df.to_csv(f"{ticker}_news.csv", index=False)
    print(f"{ticker}: {len(df)} articles")

