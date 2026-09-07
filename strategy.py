import pandas as pd

def moving_average_signal(prices, short_window, long_window):
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    mov_avg_df = (short_ma>long_ma).astype(int)
    return mov_avg_df
