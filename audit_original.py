"""Reproduce the archived original forecast code without import side effects."""
import ast
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from project_config import ROOT, OUTPUT, TICKERS


def main():
    with zipfile.ZipFile(OUTPUT / "original_source_before_corrections.zip") as archive:
        source = archive.read("forecast_comparision.py").decode("utf-8-sig")
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    ns = {"np": np, "pd": pd, "minimize": minimize, "norm": norm}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "archived_original", "exec"), ns)
    records, coverage, fits = [], [], {}
    for ticker in TICKERS:
        data = pd.read_csv(ROOT / f"{ticker}_merged_lagged.csv", index_col=0, parse_dates=True)
        news = pd.read_csv(ROOT / f"{ticker}_sentiment.csv")
        dates = pd.to_datetime(news.date, utc=True)
        coverage.append({"ticker": ticker, "articles": len(news),
                         "duplicate_date_headline": int(news.duplicated(["date", "headline"]).sum()),
                         "news_first": str(dates.min()), "news_last": str(dates.max()),
                         "weekend_pct": 100 * float(dates.dt.dayofweek.ge(5).mean()),
                         "zero_lag_pct": 100 * float(data.sentiment_lag1.eq(0).mean()),
                         "price_rows": len(data)})
        r, s = data.log_return.to_numpy()*100, data.sentiment_lag1.to_numpy()
        split = int(len(r)*.8)
        g, gx = ns["fit_garch"](r[:split]), ns["fit_garch_x"](r[:split],s[:split])
        vg, vx = ns["garch_variance"](g.x,r), ns["garch_x_variance"](gx.x,r,s)
        fits[ticker] = {"garch": g.x.tolist(), "garch_x": gx.x.tolist(), "garch_nll": float(g.fun), "gx_nll": float(gx.fun)}
        if not(g.success and gx.success) or vg is None or vx is None:
            raise RuntimeError(f"Original fit failed for {ticker}")
        y=(r[split:]-g.x[0])**2
        for loss in ["QLIKE", "MSE"]:
            a=ns["qlike_loss"](y,vg[split:]) if loss=="QLIKE" else (y-vg[split:])**2
            b=ns["qlike_loss"](y,vx[split:]) if loss=="QLIKE" else (y-vx[split:])**2
            dm,p,diff=ns["diebold_mariano"](a,b)
            records.append({"ticker":ticker,"loss":loss,"n_train":split,"n_test":len(y),
                            "test_start":str(data.index[split].date()),"test_end":str(data.index[-1].date()),
                            "mean_garch":float(a.mean()),"mean_garch_x":float(b.mean()),"difference":diff,"dm_original":dm,"p_original":p})
        print(ticker,"original reproduced",flush=True)
    pd.DataFrame(records).to_csv(OUTPUT/"original_metrics.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUTPUT/"data_audit.csv",index=False)
    (OUTPUT/"original_fits.json").write_text(json.dumps(fits,indent=2))


if __name__=="__main__":
    main()
