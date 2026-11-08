"""Explicit optional rescoring; existing sentiment caches are never overwritten."""
import ast
import json
import pandas as pd
from project_config import ROOT, TICKERS, OUTPUT
from sentiusingpytorch import get_sentiment_batch, load_model


def main():
    directory = OUTPUT / "rescored_sentiment"
    directory.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        destination = directory / f"{ticker}_sentiment.csv"
        if destination.exists():
            raise FileExistsError(destination)
        df = pd.read_csv(ROOT / f"{ticker}_news.csv")
        df["symbols"] = df.symbols.map(ast.literal_eval)
        df = df.loc[df.symbols.map(lambda symbols:ticker in symbols)].copy()
        df = df.drop_duplicates(["date","headline"])
        scores = []
        for i in range(0,len(df),64):
            scores.extend(get_sentiment_batch(df.headline.iloc[i:i+64].tolist()))
        df["sentiment"] = scores
        df.to_csv(destination,index=False)
        print(ticker,len(df),flush=True)
    _, model, device = load_model()
    (directory/"model_metadata.json").write_text(json.dumps({"model":"ProsusAI/finbert",
        "revision":getattr(model.config,"_commit_hash",None),"device":device,"id2label":model.config.id2label},indent=2))


if __name__ == "__main__":
    main()
