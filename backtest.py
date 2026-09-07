import pandas as pd

def run_backtest(prices, signal, transaction_cost_bps=10):
    daily_returns = prices.pct_change()
    lagged_signal = signal.shift(1).fillna(0)
    row_sum = lagged_signal.sum(axis=1)
    position_weights = lagged_signal.div(row_sum.replace(0,1), axis=0)
    strat_return = (position_weights * daily_returns).sum(axis=1)
    position_changes = lagged_signal.diff().abs().sum(axis=1)
    costs = position_changes * (transaction_cost_bps / 10000)
    transcost_return = strat_return - costs
    strategy_equity_curve = (1 + transcost_return).cumprod()
    benchmark_return = daily_returns.mean(axis=1)
    benchmark_equity_curve = (1 + benchmark_return).cumprod()
    result = pd.DataFrame({
        "strategy_return": strat_return,
        "strategy_equity": strategy_equity_curve,
        "benchmark_return": benchmark_return,
        "benchmark_equity": benchmark_equity_curve,
        "turnover": position_changes
    })
    return result.dropna()
