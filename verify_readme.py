"""
verify_readme_figures.py — re-runs the actual pipeline and checks every
number quoted in README.md against a fresh computation.
"""
import pandas as pd

from data import load_prices, load_synthetic_prices
from strategy import moving_average_signal, mean_reversion_signal
from backtest import run_backtest
from metrics import summarise

TOLERANCE = 0.01  # 1 percentage point of slack, to allow for minor float/data drift

# ---------------------------------------------------------------------------
# Every number currently claimed in README.md, as (value, tolerance_override)
# Percentages are stored as decimals (34.2% -> 0.342) to match summarise()'s output.
# ---------------------------------------------------------------------------

EXPECTED = {
    "base": {
        "strategy_total_return": 3.420,
        "strategy_annualised_return": 0.161,
        "strategy_sharpe_ratio": 0.85,
        "strategy_max_drawdown": -0.287,
        "benchmark_total_return": 8.234,
        "benchmark_annualised_return": 0.250,
        "benchmark_sharpe_ratio": 1.07,
        "benchmark_max_drawdown": -0.379,
    },
    "cost_0": {"strategy_total_return": 4.38, "strategy_sharpe_ratio": 0.95, "strategy_max_drawdown": -0.262},
    "cost_10": {"strategy_total_return": 3.42, "strategy_sharpe_ratio": 0.85, "strategy_max_drawdown": -0.287},
    "cost_30": {"strategy_total_return": 1.99, "strategy_sharpe_ratio": 0.65, "strategy_max_drawdown": -0.341},
    "cost_50": {"strategy_total_return": 1.01, "strategy_sharpe_ratio": 0.45, "strategy_max_drawdown": -0.396},
    "params_10_30": {"strategy_sharpe_ratio": 0.66, "strategy_total_return": 2.00, "strategy_max_drawdown": -0.527},
    "params_20_50": {"strategy_sharpe_ratio": 0.85, "strategy_total_return": 3.42, "strategy_max_drawdown": -0.287},
    "params_20_80": {"strategy_sharpe_ratio": 0.90, "strategy_total_return": 3.96, "strategy_max_drawdown": -0.452},
    "params_30_100": {"strategy_sharpe_ratio": 1.16, "strategy_total_return": 7.27, "strategy_max_drawdown": -0.298},
    "params_40_120": {"strategy_sharpe_ratio": 1.15, "strategy_total_return": 7.91, "strategy_max_drawdown": -0.279},
    "params_50_150": {"strategy_sharpe_ratio": 1.10, "strategy_total_return": 6.74, "strategy_max_drawdown": -0.287},
    "mean_reversion": {"strategy_sharpe_ratio": 0.68, "strategy_total_return": 2.53, "strategy_max_drawdown": -0.299},
    "breaker_off": {"strategy_total_return": 3.42, "strategy_max_drawdown": -0.287, "strategy_sharpe_ratio": 0.85},
    "breaker_on": {"strategy_total_return": 6.325, "strategy_max_drawdown": -0.200, "strategy_sharpe_ratio": 1.19},
}

TICKERS = ["MSFT", "AMZN", "JPM", "GOOGL"]
START, END = "2016-01-01", "2026-01-01"


def check(label, actual, expected, tol=TOLERANCE):
    passed = True
    for key, expected_val in expected.items():
        actual_val = actual.get(key)
        if actual_val is None:
            print(f"  [{label}] MISSING KEY: {key}")
            passed = False
            continue
        diff = abs(actual_val - expected_val)
        status = "PASS" if diff <= tol else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  [{label}] {key:30s} README={expected_val:>8.3f}  actual={actual_val:>8.3f}  diff={diff:.4f}  {status}")
    return passed


def main():
    print(f"Loading real price data for {TICKERS} ({START} to {END})...\n")
    prices = load_prices(TICKERS, start=START, end=END)

    all_passed = True

    # --- Base result (20/50, 10 bps) ---
    signal = moving_average_signal(prices, 20, 50)
    result = summarise(run_backtest(prices, signal, 10))
    all_passed &= check("base", result, EXPECTED["base"])

    # --- Cost sensitivity ---
    for bps, key in [(0, "cost_0"), (10, "cost_10"), (30, "cost_30"), (50, "cost_50")]:
        result = summarise(run_backtest(prices, signal, bps))
        all_passed &= check(key, result, EXPECTED[key])

    # --- Parameter sensitivity ---
    for short, long, key in [
        (10, 30, "params_10_30"), (20, 50, "params_20_50"), (20, 80, "params_20_80"),
        (30, 100, "params_30_100"), (40, 120, "params_40_120"), (50, 150, "params_50_150"),
    ]:
        sig = moving_average_signal(prices, short, long)
        result = summarise(run_backtest(prices, sig, 10))
        all_passed &= check(key, result, EXPECTED[key])

    # --- Mean reversion ---
    mr_signal = mean_reversion_signal(prices)
    mr_result = summarise(run_backtest(prices, mr_signal, 10))
    all_passed &= check("mean_reversion", mr_result, EXPECTED["mean_reversion"])

    # --- Drawdown breaker ---
    off_result = summarise(run_backtest(prices, signal, 10, limit=None))
    on_result = summarise(run_backtest(prices, signal, 10, limit=-0.20))
    all_passed &= check("breaker_off", off_result, EXPECTED["breaker_off"])
    all_passed &= check("breaker_on", on_result, EXPECTED["breaker_on"])

    print()
    if all_passed:
        print("ALL README FIGURES MATCH the current codebase. Safe to publish.")
    else:
        print("SOME FIGURES DO NOT MATCH — update the README or investigate why the code changed.")


if __name__ == "__main__":
    main()