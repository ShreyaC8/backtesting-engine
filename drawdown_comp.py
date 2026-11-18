"""
drawdown_comparison.py — quick script to get the exact numbers needed
for the README's drawdown circuit breaker section.

Run with: python drawdown_comparison.py
"""
from data import load_prices
from strategy import moving_average_signal
from backtest import run_backtest
from metrics import summarise

prices = load_prices(["MSFT", "AMZN", "JPM", "GOOGL"], start="2016-01-01", end="2026-01-01")
signal = moving_average_signal(prices, short_window=20, long_window=50)

no_breaker = summarise(run_backtest(prices, signal, transaction_cost_bps=10, limit=None))
with_breaker = summarise(run_backtest(prices, signal, transaction_cost_bps=10, limit=-0.20))

print("WITHOUT circuit breaker:")
print(f"  Total return:  {no_breaker['strategy_total_return']:.1%}")
print(f"  Max drawdown:  {no_breaker['strategy_max_drawdown']:.1%}")
print(f"  Sharpe:        {no_breaker['strategy_sharpe_ratio']:.2f}")
print()
print("WITH -20% circuit breaker:")
print(f"  Total return:  {with_breaker['strategy_total_return']:.1%}")
print(f"  Max drawdown:  {with_breaker['strategy_max_drawdown']:.1%}")
print(f"  Sharpe:        {with_breaker['strategy_sharpe_ratio']:.2f}")