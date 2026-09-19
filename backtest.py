import pandas as pd

def apply_drawdown_limit(equity_curve, weights, limit=-0.20):
    adjusted_weights = weights.copy()
    peak = 1.0
    for date in equity_curve.index:
        peak = max(peak, equity_curve.loc[date])
        current_drawdown = equity_curve.loc[date] / peak - 1
        if current_drawdown < limit:
            adjusted_weights.loc[date] = 0   # force to cash
    return adjusted_weights

def run_backtest(prices, signal, transaction_cost_bps=10, limit=None):
    daily_returns = prices.pct_change()
    lagged_signal = signal.shift(1).fillna(0)
    row_sum = lagged_signal.sum(axis=1)
    position_weights = lagged_signal.div(row_sum.replace(0,1), axis=0)

    if limit is not None:
        equity_result = run_backtest(prices, signal, transaction_cost_bps, None)
        equity_arg = equity_result["strategy_equity"]
        adjusted_weights = apply_drawdown_limit(equity_arg, position_weights, limit)
        position_weights = adjusted_weights

    strat_return = (position_weights * daily_returns).sum(axis=1)
    position_changes = lagged_signal.diff().abs().sum(axis=1)
    costs = position_changes * (transaction_cost_bps / 10000)

    transcost_return = strat_return - costs
    strategy_equity_curve = (1 + transcost_return).cumprod()
    benchmark_return = daily_returns.mean(axis=1)
    benchmark_equity_curve = (1 + benchmark_return).cumprod()

    result = pd.DataFrame({
        "strategy_return": transcost_return,
        "strategy_equity": strategy_equity_curve,
        "benchmark_return": benchmark_return,
        "benchmark_equity": benchmark_equity_curve,
        "turnover": position_changes
    })
    return result.dropna()
