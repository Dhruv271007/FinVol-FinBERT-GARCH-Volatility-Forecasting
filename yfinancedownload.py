"""Price data: summarise the cached prices, or explicitly download a fresh copy.

    python yfinancedownload.py              # describe cached price_data.csv
    python yfinancedownload.py --download   # fetch with yfinance into output/refreshed_prices/

No network call happens at import, and the cached price_data.csv is never overwritten.
"""
import argparse
from project_config import TICKERS, OUTPUT


def download(start="2020-06-05", end="2026-06-05"):
    import yfinance as yf
    destination = OUTPUT / "refreshed_prices" / "price_data.csv"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = yf.download(list(TICKERS), start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("yfinance returned no data")
    data.to_csv(destination)
    print(destination, data.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="fetch a fresh price snapshot")
    args = parser.parse_args()
    if args.download:
        download()
        return
    from corrected_data import load_prices
    _, _, rolling_vol = load_prices()
    print(rolling_vol.describe())


if __name__ == "__main__":
    main()
