import pandas as pd
import matplotlib.pyplot as plt

from data import load_prices, load_synthetic_prices
from strategy import moving_average_signal
from backtest import run_backtest
from metrics import summarise

def main():
    prices = load_prices(["MSFT", "AMZN", "JPM", "GOOGL"], start="2016-01-01", end="2026-01-01")
    signal = moving_average_signal(prices, short_window=20, long_window=50)
    result = run_backtest(prices, signal, transaction_cost_bps=10)
    metric_summary = summarise(result)
    print(metric_summary)
    x1 = result["strategy_equity"].index
    y1 = result["strategy_equity"]
    x2 = result["benchmark_equity"].index
    y2 = result["benchmark_equity"]
    plt.plot(x1,y1,'r')
    plt.plot(x2,y2,'b')
    plt.legend(["Strategy Equity", "Benchmark Equity"])
    plt.show()

if __name__ == "__main__":
    main()