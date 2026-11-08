"""Plot corrected sentiment against squared returns for descriptive inspection."""
import pandas as pd
import matplotlib.pyplot as plt
from project_config import TICKERS, CORRECTED_DATA, EXCLUDED

def main():
    for ticker in TICKERS:
        if ticker in EXCLUDED:
            continue
        df = pd.read_csv(CORRECTED_DATA / f"{ticker}_merged_lagged.csv")
        df = df.loc[df.coverage_envelope.astype(str).str.lower().eq("true")]
        fig, ax = plt.subplots()
        ax.scatter(df.sentiment_lag1, (100*df.log_return)**2, alpha=.3, s=8)
        ax.set(xlabel="Lagged mean signed sentiment", ylabel="Squared percentage log return", title=ticker)
        plt.show()

if __name__ == "__main__":
    main()
