"""Explicit optional download using the installed Alpaca SDK's token pagination.

No network call is made at import. A new snapshot is written under output/;
cached historical inputs remain unchanged. Run only when a refresh is intended.
"""
from datetime import datetime, timezone
import pandas as pd
from project_config import TICKERS, OUTPUT


def main():
    import os
    from dotenv import load_dotenv
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest
    from alpaca.common.enums import Sort
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set Alpaca credentials in your local environment")
    client = NewsClient(api_key=key, secret_key=secret)
    destination = OUTPUT / "refreshed_news"
    destination.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        path = destination / f"{ticker}_news.csv"
        if path.exists():
            raise FileExistsError(path)
        # limit=None lets the SDK follow next_page_token through every page.
        request = NewsRequest(symbols=ticker, start=datetime(2020,6,5,tzinfo=timezone.utc),
                              end=datetime(2026,6,5,tzinfo=timezone.utc), sort=Sort.ASC, limit=None)
        items = client.get_news(request).data["news"]
        retrieved_at = datetime.now(timezone.utc).isoformat()
        records = [{"id":a.id,"date":a.created_at,"updated_at":a.updated_at,"retrieved_at":retrieved_at,
                    "headline":a.headline,"symbols":a.symbols} for a in items]
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.drop_duplicates("id").sort_values("date")
        frame.to_csv(path,index=False)
        print(ticker,len(frame),flush=True)


if __name__ == "__main__":
    main()
