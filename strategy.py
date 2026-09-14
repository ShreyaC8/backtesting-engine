import pandas as pd

def moving_average_signal(prices, short_window=20, long_window=50):
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    mov_avg_df = (short_ma>long_ma).astype(int)
    return mov_avg_df

def mean_reversion_signal(prices, window=20, entry_threshold=-1.0, exit_threshold=0.0):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    z_score = ((prices - rolling_mean) / rolling_std).fillna(0)
    signal = pd.DataFrame(columns=list(z_score.columns), index=prices.index)
    for label, values in z_score.items():
        signal_vals = []
        for row in values:
            if row < entry_threshold:
                signal_vals.append(1)
            elif row > exit_threshold or not signal_vals:
                signal_vals.append(0)
            else:
                signal_vals.append(signal_vals[-1])
        signal[label] = signal_vals
    return signal