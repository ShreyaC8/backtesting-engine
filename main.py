import pandas as pd
import matplotlib.pyplot as plt

from data import load_prices, load_synthetic_prices
from strategy import moving_average_signal
from backtest import run_backtest
from metrics import summarise

def cost_sensitivity_analysis(prices):
    signal = moving_average_signal(prices, short_window=20, long_window=50)

    result_0 = run_backtest(prices, signal, 0)
    metric_summary_0 = summarise(result_0)
    print("Summary for transaction cost 0 bps:")
    print(metric_summary_0)
    print("\n")

    result_10 = run_backtest(prices, signal, 10)
    metric_summary_10 = summarise(result_10)
    print("Summary for transaction cost 10 bps:")
    print(metric_summary_10)
    print("\n")

    result_30 = run_backtest(prices, signal, 30)
    metric_summary_30 = summarise(result_30)
    print("Summary for transaction cost 30 bps:")
    print(metric_summary_30)
    print("\n")

    result_50 = run_backtest(prices, signal, 50)
    metric_summary_50 = summarise(result_50)
    print("Summary for transaction cost 50 bps:")
    print(metric_summary_50)
    print("\n")
    
    cost_sensitive_analysis, [[cost_0, cost_10], [cost_30, cost_50]] = plt.subplots(2, 2, figsize=(10, 4))

    cost_0.plot(result_0["strategy_equity"].index, result_0["strategy_equity"], label="Strategy Equity", color="blue")
    cost_0.plot(result_0["benchmark_equity"].index, result_0["benchmark_equity"], label="Benchmark Equity", color="red")
    cost_0.set_title("Transaction Cost (bps): 0")
    cost_0.legend()

    cost_10.plot(result_10["strategy_equity"].index, result_10["strategy_equity"], label="Strategy Equity", color="blue")
    cost_10.plot(result_10["benchmark_equity"].index, result_10["benchmark_equity"], label="Benchmark Equity", color="red")
    cost_10.set_title("Transaction Cost (bps): 10")
    cost_10.legend()

    cost_30.plot(result_30["strategy_equity"].index, result_30["strategy_equity"], label="Strategy Equity", color="blue")
    cost_30.plot(result_30["benchmark_equity"].index, result_30["benchmark_equity"], label="Benchmark Equity", color="red")
    cost_30.set_title("Transaction Cost (bps): 30")
    cost_30.legend()

    cost_50.plot(result_50["strategy_equity"].index, result_50["strategy_equity"], label="Strategy Equity", color="blue")
    cost_50.plot(result_50["benchmark_equity"].index, result_50["benchmark_equity"], label="Benchmark Equity", color="red")
    cost_50.set_title("Transaction Cost (bps): 50")
    cost_50.legend()

    cost_sensitive_analysis.suptitle("20/50-Day MA Crossover Strategy vs. Buy & Hold (2016–2026)")
    cost_sensitive_analysis.supxlabel("Date")
    cost_sensitive_analysis.supylabel("Account Equity")

    plt.tight_layout()
    plt.show()
    plt.savefig("output/cost_sensitive_analysis_equity_curves.png")
    plt.show()

def parameter_sensitivity(prices, window_pairs):
    df_basis = []
    for short_window, long_window in window_pairs:
        key_metrics = {}
        signal = moving_average_signal(prices, short_window, long_window)
        result = run_backtest(prices, signal, 10)
        metric_summary = summarise(result)
        key_metrics["Short/Long"] = str(short_window)+"/"+str(long_window)
        key_metrics["Sharpe"] = metric_summary["strategy_sharpe_ratio"]
        key_metrics["Total Return"] = metric_summary["strategy_total_return"]
        key_metrics["Max Drawdown"] = metric_summary["strategy_max_drawdown"]
        df_basis.append(key_metrics)
    table = pd.DataFrame(df_basis)
    print(table)

    x = list(table["Short/Long"])
    y = list(table["Sharpe"])
    plt.bar(x,y)
    plt.show()

def main():
    window_pairs = [(10, 30), (20, 50), (20, 80), (30, 100), (40, 120), (50, 150)]
    prices = load_prices(["MSFT", "AMZN", "JPM", "GOOGL"], start="2016-01-01", end="2026-01-01")
    parameter_sensitivity(prices,window_pairs)

if __name__ == "__main__":
    main()