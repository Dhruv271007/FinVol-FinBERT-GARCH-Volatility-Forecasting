# Volatility Forecasting with FinBERT Sentiment

## Overview
Predicting next-day realized volatility for top 10 S&P 500 
stocks using FinBERT sentiment scores from financial news,
benchmarked against GARCH(1,1) via Diebold-Mariano test.

## Pipeline
1. Price data via yfinance
2. 30-day realized volatility
3. News headlines via Alpaca Markets API
4. FinBERT sentiment scoring (GPU-accelerated)
5. GARCH-X model with lagged sentiment

## Setup
pip install -r requirements.txt
Add API keys to .env file (see .env.example)

## Status
Work in progress
