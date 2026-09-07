import yfinance as yf
import pandas as pd
import numpy as np

def load_prices(ticker_list, start, end):
    data = yf.download(tickers = ticker_list, start = start, end = end)["Close"]
    if len(ticker_list) == 1:
        close_df = pd.Series(data)
    else:
        close_df = pd.DataFrame(data)
    close_df.dropna(how="all", inplace=True)
    return close_df

def load_synthetic_prices(tickers, start, end, seed):
    dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(seed=seed)
    drift = 0.0002
    vol = 0.015
    data = {}
    for ticker in tickers:
        daily_returns = pd.Series(rng.normal(drift, vol, len(dates)), index=dates)
        price = 100 * (1 + daily_returns).cumprod()
        data[ticker] = price
    df = pd.DataFrame(data, index=dates)
    return df
