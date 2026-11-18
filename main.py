import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_prices, load_synthetic_prices
from strategy import moving_average_signal, mean_reversion_signal
from backtest import run_backtest
from metrics import summarise
from itertools import product

UNIVERSES = {
    "commodities": ["BZ=F", "NG=F"],
    "equities": ["MSFT", "AMZN", "JPM", "GOOGL"],
}

NAMES = [
    "ma_curve.png", "mr_curve.png", "ma_vs_mr_curve.png",
    "cost_sensitive_analysis_curves.png", "parameter_sensitivity_curves.png",
]
EXPECTED_FILES = [f"output/{u}/{n}" for u in ("commodities", "equities") for n in NAMES]

def moving_average(prices, short_window=20, long_window=50, limit=None, out_dir="output"):
    plt.figure()
    signal = moving_average_signal(prices, short_window, long_window)
    result = run_backtest(prices, signal, 10, limit)
    metric_summary = summarise(result)
    print("Summary for Moving Average strategy:")
    print(metric_summary)
    print("\n")

    x1 = result["strategy_equity"].index
    y1 = result["strategy_equity"]
    x2 = result["benchmark_equity"].index
    y2 = result["benchmark_equity"]

    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'b')
    plt.legend(["Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Equity/ Commodity")
    plt.title("MA Crossover Strategy vs. Buy & Hold")
    plt.savefig(f"{out_dir}/ma_curve.png")
    plt.close()

def mean_reversion(prices, window=20, entry_threshold=-1.0, exit_threshold=0.0, limit=None, out_dir="output"):
    plt.figure()
    signal = mean_reversion_signal(prices, window, entry_threshold, exit_threshold)
    result = run_backtest(prices, signal, 10, limit)
    metric_summary = summarise(result)
    print("Summary for Mean Reversion strategy:")
    print(metric_summary)
    print("\n")

    x1 = result["strategy_equity"].index
    y1 = result["strategy_equity"]
    x2 = result["benchmark_equity"].index
    y2 = result["benchmark_equity"]

    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'b')
    plt.legend(["Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Equity/ Commodity")
    plt.title("MR Crossover Strategy vs. Buy & Hold")
    plt.savefig(f"{out_dir}/mr_curve.png")
    plt.close()

def cost_sensitivity_analysis(prices, out_dir="output"):
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

    # plt.subplots() always creates a brand-new figure, so no plt.figure() needed here
    cost_sensitive_analysis, [[cost_0, cost_10], [cost_30, cost_50]] = plt.subplots(2, 2, figsize=(10, 4))

    cost_0.plot(result_0["strategy_equity"].index, result_0["strategy_equity"], label="Strategy", color="blue")
    cost_0.plot(result_0["benchmark_equity"].index, result_0["benchmark_equity"], label="Benchmark", color="red")
    cost_0.set_title("Transaction Cost (bps): 0")
    cost_0.legend()

    cost_10.plot(result_10["strategy_equity"].index, result_10["strategy_equity"], label="Strategy", color="blue")
    cost_10.plot(result_10["benchmark_equity"].index, result_10["benchmark_equity"], label="Benchmark", color="red")
    cost_10.set_title("Transaction Cost (bps): 10")
    cost_10.legend()

    cost_30.plot(result_30["strategy_equity"].index, result_30["strategy_equity"], label="Strategy", color="blue")
    cost_30.plot(result_30["benchmark_equity"].index, result_30["benchmark_equity"], label="Benchmark", color="red")
    cost_30.set_title("Transaction Cost (bps): 30")
    cost_30.legend()

    cost_50.plot(result_50["strategy_equity"].index, result_50["strategy_equity"], label="Strategy", color="blue")
    cost_50.plot(result_50["benchmark_equity"].index, result_50["benchmark_equity"], label="Benchmark", color="red")
    cost_50.set_title("Transaction Cost (bps): 50")
    cost_50.legend()

    cost_sensitive_analysis.suptitle("20/50-Day MA Crossover Strategy vs. Buy & Hold (2016–2026)")
    cost_sensitive_analysis.supxlabel("Date")
    cost_sensitive_analysis.supylabel("Account Equity/ Commodity")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/cost_sensitive_analysis_curves.png")
    plt.close()

def parameter_sensitivity(prices, window_pairs, out_dir="output"):
    df_basis = []
    for short_window, long_window in window_pairs:
        key_metrics = {}
        signal = moving_average_signal(prices, short_window, long_window)
        result = run_backtest(prices, signal, 10)
        metric_summary = summarise(result)
        key_metrics["Short/Long"] = str(short_window) + "/" + str(long_window)
        key_metrics["Sharpe"] = metric_summary["strategy_sharpe_ratio"]
        key_metrics["Total Return"] = metric_summary["strategy_total_return"]
        key_metrics["Max Drawdown"] = metric_summary["strategy_max_drawdown"]
        df_basis.append(key_metrics)
    table = pd.DataFrame(df_basis)
    print("Parameter sensitivity results:")
    print(table)
    print("\n")

    plt.figure()
    x = list(table["Short/Long"])
    y = list(table["Sharpe"])
    plt.bar(x, y)
    plt.xlabel("Short/Long")
    plt.ylabel("Sharpe")
    plt.title("Sharpe Ratio by Moving Average Window Pair")
    plt.savefig(f"{out_dir}/parameter_sensitivity_curves.png")
    plt.close()

def strategy_comparison(prices, out_dir="output"):
    moving_avg_sig = moving_average_signal(prices)
    mean_reversion_sig = mean_reversion_signal(prices)

    moving_avg_result = run_backtest(prices, moving_avg_sig, 10)
    mean_reversion_result = run_backtest(prices, mean_reversion_sig, 10)

    moving_summary = summarise(moving_avg_result)
    mean_summary = summarise(mean_reversion_result)

    basis = [{
        "Strategy": "Moving Average",
        "Sharpe": moving_summary["strategy_sharpe_ratio"],
        "Total Return": moving_summary["strategy_total_return"],
        "Max Drawdown": moving_summary["strategy_max_drawdown"]
        },
        {
        "Strategy": "Mean Reversion",
        "Sharpe": mean_summary["strategy_sharpe_ratio"],
        "Total Return": mean_summary["strategy_total_return"],
        "Max Drawdown": mean_summary["strategy_max_drawdown"]
        }]

    comparison = pd.DataFrame(basis)
    print("Strategy comparison:")
    print(comparison)
    print("\n")

    x1 = moving_avg_result["strategy_equity"].index
    y1 = moving_avg_result["strategy_equity"]
    x2 = mean_reversion_result["strategy_equity"].index
    y2 = mean_reversion_result["strategy_equity"]
    x3 = moving_avg_result["benchmark_equity"].index
    y3 = moving_avg_result["benchmark_equity"]

    plt.figure()
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'b')
    plt.plot(x3, y3, 'g')
    plt.legend(["MA Strategy", "MR Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Equity/ Commodity")
    plt.title("MA Crossover Strategy vs. MR Strategy")
    plt.savefig(f"{out_dir}/ma_vs_mr_curve.png")
    plt.close()

def out_of_sample_test(prices):
    train = ("2016-01-01", "2020-12-31")
    test = ("2021-01-01", "2026-01-01")

    def run_period(signal, period, cost=10):
        start, end = period
        return summarise(run_backtest(prices.loc[start:end], signal.loc[start:end], cost))

    best = None
    for window, entry in product([10, 20, 40], [-0.5, -1.0, -1.5]):
        sig = mean_reversion_signal(prices, window, entry, 0.0)
        sharpe = run_period(sig, train)["strategy_sharpe_ratio"]
        print(f"window={window}, entry={entry}, train Sharpe={sharpe:.2f}")
        if best is None or sharpe > best[0]:
            best = (sharpe, window, entry)

    _, window, entry = best
    print(f"\nChosen on 2016-2020: window={window}, entry={entry}")
    sig = mean_reversion_signal(prices, window, entry, 0.0)
    print("Train (2016-2020):", run_period(sig, train))
    print("Test  (2021-2026):", run_period(sig, test))

def run_universe(name, tickers):
    print(f"\n{'=' * 20} {name.upper()} {'=' * 20}\n")
    out_dir = f"output/{name}"
    os.makedirs(out_dir, exist_ok=True)  # savefig fails if the folder doesn't exist

    window_pairs = [(10, 30), (20, 50), (20, 80), (30, 100), (40, 120), (50, 150)]
    prices = load_prices(tickers, start="2016-01-01", end="2026-01-01")
    assert (prices > 0).all().all(), "Non-positive price in data"

    moving_average(prices, out_dir=out_dir)
    mean_reversion(prices, limit=-0.2, out_dir=out_dir)
    cost_sensitivity_analysis(prices, out_dir=out_dir)
    parameter_sensitivity(prices, window_pairs, out_dir=out_dir)
    strategy_comparison(prices, out_dir=out_dir)
    out_of_sample_test(prices)

def main():
    for name, tickers in UNIVERSES.items():
        run_universe(name, tickers)

if __name__ == "__main__":
    main()