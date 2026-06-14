import sys
sys.stdout.reconfigure(encoding='utf-8')
print("imports starting...")
print("pandas imported")
import pandas as pd
print("ast imported")
import ast
print("sentiment importing...")
from sentiment import get_sentiment_batch, model, tokenizer, device
print("all imports done!")
tickers=["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","TSLA"]
print("starting loop...")
print(f"looking for: AAPL_news.csv")
print(f"file exists: {pd.io.common.file_exists('AAPL_news.csv')}")
for ticker in tickers:
    df = pd.read_csv(f"{ticker}_news.csv")
    df['symbols']=df['symbols'].apply(lambda x: ast.literal_eval(x))
    df = df[df['symbols'].apply(lambda x: ticker in x)]
    print(f"{ticker}: {len(df)} articles after filtering ")
    # batch processing
    headlines = df['headline'].tolist()  # convert column to python list
    sentiment_scores = []
    
    for i in range(0, len(headlines), 64):  # step through in chunks of 64
        batch = headlines[i:i+64]           # slice 64 headlines at a time
        scores = get_sentiment_batch(batch)
        sentiment_scores.extend(scores)
        print(f"  processed {min(i+64, len(headlines))}/{len(headlines)}")
    
    df['sentiment'] = sentiment_scores
    df.to_csv(f"{ticker}_sentiment.csv", index=False)
    print(f"{ticker} done!")


