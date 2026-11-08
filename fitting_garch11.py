"""Descriptive full-sample ARCH-library GARCH fit, not forecast evaluation."""
import pandas as pd
from arch import arch_model
from project_config import TICKERS, CORRECTED_DATA, EXCLUDED

def main():
    for ticker in TICKERS:
        if ticker in EXCLUDED:
            continue
        df = pd.read_csv(CORRECTED_DATA / f"{ticker}_merged_lagged.csv", index_col="Date", parse_dates=True)
        df = df.loc[df.coverage_envelope.astype(str).str.lower().eq("true")]
        model = arch_model(df.log_return * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        result = model.fit(disp="off", cov_type="robust")
        if result.convergence_flag:
            raise RuntimeError(f"{ticker}: optimizer failed")
        print(ticker)
        print(result.summary())

if __name__ == "__main__":
    main()
