"""Target-aligned news windows using XNYS closes plus five minutes.

created_at is an availability proxy, not proof of first-seen article text.
Source CSVs stay immutable. Corrected frames are written under output/.
"""
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from project_config import ROOT, TICKERS, CORRECTED_DATA, ISSUANCE_DELAY_MINUTES, EXCLUDED


def load_prices():
    data = pd.read_csv(ROOT / "price_data.csv", header=[0, 1], index_col=0, parse_dates=True).sort_index()
    if data.index.has_duplicates or (data["Close"] <= 0).any().any():
        raise ValueError("Price dates must be unique and closes positive")
    close = data["Close"]
    returns = np.log(close).diff()  # Never forward fill missing prices.
    rolling_vol = returns.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
    return data, returns, rolling_vol


def issuance_times(price_dates):
    dates = pd.DatetimeIndex(price_dates).tz_localize(None)
    calendar = xcals.get_calendar("XNYS", start=dates.min() - pd.Timedelta(days=10), end=dates.max() + pd.Timedelta(days=10))
    closes = calendar.schedule.loc[dates, "close"]  # one vectorised lookup, not a per-date call
    return pd.DatetimeIndex(closes).tz_convert("UTC") + pd.Timedelta(minutes=ISSUANCE_DELAY_MINUTES)


def load_sentiment(ticker):
    frame = pd.read_csv(ROOT / f"{ticker}_sentiment.csv")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="raise")
    if frame.headline.isna().any() or not np.isfinite(frame.sentiment).all() or frame.sentiment.abs().max() > 1 + 1e-8:
        raise ValueError(f"Invalid sentiment for {ticker}")
    return frame.sort_values("date").drop_duplicates(["date", "headline"], keep="first")


def aggregate_news(news, issues):
    # A weekend story enters Monday issuance, forecasting Tuesday's return.
    # pandas 3 can infer microseconds for parsed CSV timestamps while calendar
    # values use nanoseconds. Convert explicitly before comparing integer times.
    buckets = np.searchsorted(issues.as_unit("ns").asi8,
                              pd.DatetimeIndex(news.date).as_unit("ns").asi8, side="left")
    use = (buckets > 0) & (buckets < len(issues))
    frame = news.loc[use, ["sentiment"]].copy()
    frame["bucket"] = buckets[use]
    grouped = frame.groupby("bucket").sentiment.agg(["mean", "count"])
    means = grouped["mean"].reindex(range(len(issues)), fill_value=0).to_numpy()
    counts = grouped["count"].reindex(range(len(issues)), fill_value=0).to_numpy()
    return means, counts


def build_merged(ticker, prices=None):
    data, returns, rolling_vol = load_prices() if prices is None else prices
    issues = issuance_times(data.index)
    news = load_sentiment(ticker)
    means, counts = aggregate_news(news, issues)
    frame = pd.DataFrame(index=data.index)
    frame.index.name = "Date"
    frame["log_return"] = returns[ticker]
    frame["realized_vol"] = rolling_vol[ticker]  # Descriptive only.
    frame["sentiment_lag1"] = pd.Series(means, index=data.index).shift(1)
    frame["news_count_lag1"] = pd.Series(counts, index=data.index).shift(1)
    frame["forecast_origin"] = pd.Series(issues, index=data.index).shift(1)
    frame["news_window_start"] = pd.Series(issues, index=data.index).shift(2)
    # Envelope is an edge screen; interior completeness cannot be established.
    frame["coverage_envelope"] = ((frame.news_window_start >= news.date.min()) &
                                  (frame.forecast_origin <= news.date.max()))
    return frame.dropna(subset=["log_return", "sentiment_lag1", "news_window_start"])


def main():
    CORRECTED_DATA.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    for ticker in TICKERS:
        if ticker in EXCLUDED:
            print(f"{ticker}: excluded - {EXCLUDED[ticker]}")
            continue
        frame = build_merged(ticker, prices)
        frame.to_csv(CORRECTED_DATA / f"{ticker}_merged_lagged.csv")
        print(f"{ticker}: {len(frame)} rows, {int(frame.coverage_envelope.sum())} in news envelope")


if __name__ == "__main__":
    main()
