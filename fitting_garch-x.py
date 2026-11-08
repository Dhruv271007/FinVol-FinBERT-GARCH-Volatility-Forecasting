"""Descriptive constrained full-sample GARCH-X fit.
Boundary fits do not receive misleading unconstrained Wald p-values.
Use corrected_forecasts.py for held-out model comparisons.
"""
import pandas as pd
from project_config import TICKERS, CORRECTED_DATA, EXCLUDED
from volatility_models import fit_model

def main():
    for ticker in TICKERS:
        if ticker in EXCLUDED:
            continue
        df = pd.read_csv(CORRECTED_DATA / f"{ticker}_merged_lagged.csv")
        df = df.loc[df.coverage_envelope.astype(str).str.lower().eq("true")]
        result = fit_model(df.log_return.to_numpy()*100, df.sentiment_lag1.to_numpy())
        print(ticker, dict(zip(["mu","omega","alpha","beta","gamma"], result.x)))
        print("NLL", result.fun, "boundary", result.boundary, "converged starts", result.n_converged)

if __name__ == "__main__":
    main()
