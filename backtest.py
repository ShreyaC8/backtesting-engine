import pandas as pd

def run_backtest(prices, signal, transaction_cost_bps=10, limit=None, cooldown=20, lag=1):
    daily_returns = prices.pct_change().fillna(0)
    lagged_signal = signal.shift(lag).fillna(0)
    target = lagged_signal.div(lagged_signal.sum(axis=1).replace(0, 1), axis=0)
    cost_rate = transaction_cost_bps / 10000

    equity, peak, days_out = 1.0, 1.0, 0
    prev_w = pd.Series(0.0, index=prices.columns)
    rows = []
    for date in prices.index:
        # decision uses only yesterday's closing equity
        if limit is not None and days_out == 0 and equity / peak - 1 < limit:
            days_out = cooldown
        if days_out > 0:
            w = prev_w * 0
            days_out -= 1
            if days_out == 0:
                peak = equity  # reset the peak after the cooldown
        else:
            w = target.loc[date]
        turnover = (w - prev_w).abs().sum()
        ret = (w * daily_returns.loc[date]).sum() - turnover * cost_rate
        equity *= 1 + ret
        peak = max(peak, equity)
        rows.append((date, ret, equity, turnover))
        prev_w = w

    out = pd.DataFrame(rows, columns=["date", "strategy_return", "strategy_equity", "turnover"]).set_index("date")
    out["benchmark_return"] = daily_returns.mean(axis=1)
    out["benchmark_equity"] = (1 + out["benchmark_return"]).cumprod()
    return out.iloc[1:]