from data import load_prices
from strategy import moving_average_signal, mean_reversion_signal
from backtest import run_backtest
from metrics import summarise

UNIVERSES = {
    "commodities": ["BZ=F", "NG=F"],
    "equities": ["MSFT", "AMZN", "JPM", "GOOGL"],
}

for name, tickers in UNIVERSES.items():
    prices = load_prices(tickers, start="2016-01-01", end="2026-01-01")
    strategies = [
        ("Moving Average", moving_average_signal(prices)),
        ("Mean Reversion", mean_reversion_signal(prices)),
    ]
    for label, signal in strategies:
        for lag in (1, 0):
            s = summarise(run_backtest(prices, signal, 10, lag=lag))
            print(f"{name:12s} {label:15s} lag={lag}  "
                  f"total return={s['strategy_total_return']:.1%}  "
                  f"Sharpe={s['strategy_sharpe_ratio']:.2f}")