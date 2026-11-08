"""Fixed-parameter sequential one-day forecasts and exploratory loss comparisons."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from project_config import TICKERS, OUTPUT, CORRECTED_DATA, EXCLUDED
from volatility_models import variance_path, fit_model, qlike_loss, diebold_mariano, holm_adjust


def run_ticker(ticker, test_frac=.2, train_frac=None, screen_gap_days=None):
    if ticker in EXCLUDED:
        raise ValueError(EXCLUDED[ticker])
    df = pd.read_csv(CORRECTED_DATA / f"{ticker}_merged_lagged.csv", index_col="Date", parse_dates=True)
    df = df.loc[df.coverage_envelope.astype(str).str.lower().eq("true")].copy()
    if screen_gap_days is not None:
        from corrected_data import load_sentiment
        dates = pd.DatetimeIndex(load_sentiment(ticker).date).normalize().unique().sort_values()
        dates = dates[(dates >= df.index.min().tz_localize("UTC")) &
                      (dates <= df.index.max().tz_localize("UTC"))]
        gap_indices = np.flatnonzero(np.diff(dates.as_unit("ns").asi8) > pd.Timedelta(days=screen_gap_days).value)
        if len(gap_indices):
            new_start = dates[gap_indices[-1] + 1]
            # Keep a contiguous final segment; never concatenate returns across gaps.
            df = df.loc[pd.to_datetime(df.news_window_start, utc=True) >= new_start]
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Dates must be sorted and unique")
    fraction = 1-test_frac if train_frac is None else train_frac
    if not 0 < fraction < 1:
        raise ValueError("Invalid training fraction")
    r, s = df.log_return.to_numpy()*100, df.sentiment_lag1.to_numpy()
    split = int(len(df)*fraction)
    if split < 250 or len(df)-split < 100:
        raise ValueError("Need 250 training and 100 evaluation returns")
    tr, ts = r[:split], s[:split]
    g = fit_model(tr)
    gx = fit_model(tr, ts, baseline=g)
    vg = variance_path(g.x, r, initial_variance=g.initial_variance)
    vx = variance_path(gx.x, r, s, gx.initial_variance)
    # A common TRAINING sample mean defines the proxy for every model.
    common_mu = float(tr.mean())
    y = (r[split:]-common_mu)**2
    # RiskMetrics EWMA: ewma[t] = .94*ewma[t-1] + .06*(r[t-1]-mu)^2, ewma[0] = training variance.
    tail, _ = lfilter([.06], [1, -.94], (r[:-1]-common_mu)**2, zi=[.94*g.initial_variance])
    ewma = np.r_[g.initial_variance, tail]
    forecasts = pd.DataFrame({"realized_proxy":y, "garch":vg[split:], "garch_x":vx[split:],
                              "ewma":ewma[split:], "constant":np.repeat(g.initial_variance,len(y))}, index=df.index[split:])
    forecasts["forecast_origin"] = df.forecast_origin.iloc[split:]
    forecasts["sentiment_lag1"] = s[split:]
    rows=[]
    for loss in ["QLIKE","MSE"]:
        values={m:qlike_loss(y,forecasts[m]) if loss=="QLIKE" else (y-forecasts[m].to_numpy())**2
                for m in ["garch","garch_x","ewma","constant"]}
        stat,p,difference=diebold_mariano(values["garch"],values["garch_x"])
        rows.append({"ticker":ticker,"loss":loss,"n_train":split,"n_test":len(y),
                     "train_start":str(df.index[0].date()),"test_start":str(df.index[split].date()),"test_end":str(df.index[-1].date()),
                     **{f"mean_{k}":float(v.mean()) for k,v in values.items()},"difference_g_minus_gx":difference,
                     "dm_hac":stat,"p_exploratory":p,"hac_lags":int(np.floor(4*(len(y)/100)**(2/9)))})
    metadata={}
    for name,fit,h in [("garch",g,vg),("garch_x",gx,vx)]:
        z=(tr-fit.x[0])/np.sqrt(h[:split])
        metadata[name]={"params":fit.x.tolist(),"nll":float(fit.fun),"aic":float(2*len(fit.x)+2*fit.fun),
                        "bic":float(len(fit.x)*np.log(split)+2*fit.fun),"initial_variance":fit.initial_variance,
                        "boundary":fit.boundary,"n_starts":fit.n_starts,"n_converged":fit.n_converged,
                        "lb_z_p10":float(acorr_ljungbox(z,lags=[10]).lb_pvalue.iloc[0]),
                        "lb_z2_p10":float(acorr_ljungbox(z**2,lags=[10]).lb_pvalue.iloc[0]),
                        "arch_lm_p10":float(het_arch(z,nlags=10,result_object=False)[1]),
                        "min_variance":float(h.min()),"max_variance":float(h.max())}
    return rows,forecasts,metadata


def run_all(train_fraction=.8,label="corrected",screen_gap_days=None):
    OUTPUT.mkdir(exist_ok=True)
    records,metadata=[],{}
    for ticker in TICKERS:
        if ticker in EXCLUDED:
            continue
        rows,forecasts,fits=run_ticker(ticker,train_frac=train_fraction,screen_gap_days=screen_gap_days)
        records.extend(rows); metadata[ticker]=fits
        forecasts.to_csv(OUTPUT/f"{label}_{ticker}_forecasts.csv")
        print(f"{ticker}: {rows[0]['n_test']} forecasts; QLIKE G={rows[0]['mean_garch']:.5f}, GX={rows[0]['mean_garch_x']:.5f}",flush=True)
    table=pd.DataFrame(records)
    table["p_holm_exploratory"]=holm_adjust(table.p_exploratory.fillna(1))
    table.to_csv(OUTPUT/f"{label}_metrics.csv",index=False)
    (OUTPUT/f"{label}_fits.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-fraction",type=float,default=.8)
    parser.add_argument("--label",default="corrected")
    parser.add_argument("--screen-long-gaps",type=int,default=None)
    args=parser.parse_args()
    run_all(args.train_fraction,args.label,args.screen_long_gaps)


if __name__=="__main__":
    main()
