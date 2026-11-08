"""Compatibility entry point for the corrected forecast experiment."""
from corrected_forecasts import run_ticker, run_all, main
from volatility_models import variance_path, fit_model, qlike_loss, diebold_mariano

def garch_variance(params, returns, initial_variance=None):
    return variance_path(params, returns, initial_variance=initial_variance)

def garch_x_variance(params, returns, sentiment, initial_variance=None):
    return variance_path(params, returns, sentiment, initial_variance)

def fit_garch(returns):
    return fit_model(returns)

def fit_garch_x(returns, sentiment):
    return fit_model(returns, sentiment)

if __name__ == "__main__":
    main()
