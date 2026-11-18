"""
verify_readme.py - re-runs the actual pipeline for both universes and checks
every number quoted in README.md against a fresh computation.
"""
from itertools import product

from data import load_prices
from strategy import moving_average_signal, mean_reversion_signal
from backtest import run_backtest
from metrics import summarise

TOLERANCE = 0.01  # 1 percentage point of slack, to allow for rounding and minor data drift

START, END = "2016-01-01", "2026-01-01"
TRAIN = ("2016-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-01-01")

UNIVERSES = {
    "commodities": ["BZ=F", "NG=F"],
    "equities": ["MSFT", "AMZN", "JPM", "GOOGL"],
}

# Percentages are stored as decimals (12.4% -> 0.124) to match summarise()'s output.
EXPECTED = {
    "commodities": {
        "base": {
            "strategy_total_return": 0.124, "strategy_annualised_return": 0.012,
            "strategy_sharpe_ratio": 0.24, "strategy_max_drawdown": -0.735,
            "benchmark_total_return": 2.143, "benchmark_annualised_return": 0.122,
            "benchmark_sharpe_ratio": 0.49, "benchmark_max_drawdown": -0.634,
        },
        "cost_0": {"strategy_total_return": 0.252, "strategy_sharpe_ratio": 0.26, "strategy_max_drawdown": -0.729},
        "cost_10": {"strategy_total_return": 0.124, "strategy_sharpe_ratio": 0.24, "strategy_max_drawdown": -0.735},
        "cost_30": {"strategy_total_return": -0.094, "strategy_sharpe_ratio": 0.19, "strategy_max_drawdown": -0.747},
        "cost_50": {"strategy_total_return": -0.270, "strategy_sharpe_ratio": 0.14, "strategy_max_drawdown": -0.758},
        "params_10_30": {"strategy_sharpe_ratio": 0.55, "strategy_total_return": 3.07, "strategy_max_drawdown": -0.561},
        "params_20_50": {"strategy_sharpe_ratio": 0.24, "strategy_total_return": 0.124, "strategy_max_drawdown": -0.735},
        "params_20_80": {"strategy_sharpe_ratio": 0.13, "strategy_total_return": -0.204, "strategy_max_drawdown": -0.712},
        "params_30_100": {"strategy_sharpe_ratio": 0.19, "strategy_total_return": 0.015, "strategy_max_drawdown": -0.630},
        "params_40_120": {"strategy_sharpe_ratio": 0.11, "strategy_total_return": -0.242, "strategy_max_drawdown": -0.679},
        "params_50_150": {"strategy_sharpe_ratio": 0.08, "strategy_total_return": -0.350, "strategy_max_drawdown": -0.786},
        "mean_reversion": {"strategy_sharpe_ratio": 0.64, "strategy_total_return": 4.81, "strategy_max_drawdown": -0.659},
        "breaker_off": {"strategy_total_return": 4.81, "strategy_max_drawdown": -0.659, "strategy_sharpe_ratio": 0.64},
        "breaker_on": {"strategy_total_return": 4.24, "strategy_max_drawdown": -0.619, "strategy_sharpe_ratio": 0.65},
        "oos_train": {
            "strategy_sharpe_ratio": 0.91, "strategy_total_return": 2.69, "strategy_max_drawdown": -0.588,
            "benchmark_sharpe_ratio": 0.44, "benchmark_total_return": 0.61,
        },
        "oos_test": {
            "strategy_sharpe_ratio": 0.60, "strategy_total_return": 1.20, "strategy_max_drawdown": -0.668,
            "benchmark_sharpe_ratio": 0.53, "benchmark_total_return": 0.86,
        },
    },
    "equities": {
        "base": {
            "strategy_total_return": 3.695, "strategy_annualised_return": 0.168,
            "strategy_sharpe_ratio": 0.88, "strategy_max_drawdown": -0.283,
            "benchmark_total_return": 8.234, "benchmark_annualised_return": 0.250,
            "benchmark_sharpe_ratio": 1.07, "benchmark_max_drawdown": -0.379,
        },
        "cost_0": {"strategy_total_return": 4.377, "strategy_sharpe_ratio": 0.95, "strategy_max_drawdown": -0.262},
        "cost_10": {"strategy_total_return": 3.695, "strategy_sharpe_ratio": 0.88, "strategy_max_drawdown": -0.283},
        "cost_30": {"strategy_total_return": 2.578, "strategy_sharpe_ratio": 0.74, "strategy_max_drawdown": -0.324},
        "cost_50": {"strategy_total_return": 1.725, "strategy_sharpe_ratio": 0.61, "strategy_max_drawdown": -0.369},
        "params_10_30": {"strategy_sharpe_ratio": 0.73, "strategy_total_return": 2.387, "strategy_max_drawdown": -0.520},
        "params_20_50": {"strategy_sharpe_ratio": 0.88, "strategy_total_return": 3.695, "strategy_max_drawdown": -0.283},
        "params_20_80": {"strategy_sharpe_ratio": 0.92, "strategy_total_return": 4.162, "strategy_max_drawdown": -0.449},
        "params_30_100": {"strategy_sharpe_ratio": 1.17, "strategy_total_return": 7.518, "strategy_max_drawdown": -0.296},
        "params_40_120": {"strategy_sharpe_ratio": 1.17, "strategy_total_return": 8.168, "strategy_max_drawdown": -0.279},
        "params_50_150": {"strategy_sharpe_ratio": 1.11, "strategy_total_return": 6.899, "strategy_max_drawdown": -0.287},
        "mean_reversion": {"strategy_sharpe_ratio": 0.73, "strategy_total_return": 2.981, "strategy_max_drawdown": -0.282},
        "breaker_off": {"strategy_total_return": 2.981, "strategy_max_drawdown": -0.282, "strategy_sharpe_ratio": 0.73},
        "breaker_on": {"strategy_total_return": 2.178, "strategy_max_drawdown": -0.305, "strategy_sharpe_ratio": 0.67},
        "oos_train": {
            "strategy_sharpe_ratio": 1.17, "strategy_total_return": 2.458, "strategy_max_drawdown": -0.285,
            "benchmark_sharpe_ratio": 1.19, "benchmark_total_return": 2.537,
        },
        "oos_test": {
            "strategy_sharpe_ratio": 0.39, "strategy_total_return": 0.373, "strategy_max_drawdown": -0.375,
            "benchmark_sharpe_ratio": 0.97, "benchmark_total_return": 1.639,
        },
    },
}


# Results with the execution lag removed (lag=0), quoted in the README's lag table
NOLAG_EXPECTED = {
    "commodities": {
        "nolag_ma": {"strategy_total_return": 0.803, "strategy_sharpe_ratio": 0.35},
        "nolag_mr": {"strategy_total_return": -1.0, "strategy_sharpe_ratio": -2.32},
    },
    "equities": {
        "nolag_ma": {"strategy_total_return": 6.827, "strategy_sharpe_ratio": 1.16},
        "nolag_mr": {"strategy_total_return": -0.989, "strategy_sharpe_ratio": -1.85},
    },
}


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


def run_period(prices, signal, period, cost=10):
    start, end = period
    return summarise(run_backtest(prices.loc[start:end], signal.loc[start:end], cost))


def verify_universe(name, tickers):
    expected = EXPECTED[name]
    print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
    print(f"Loading price data for {tickers} ({START} to {END})...\n")
    prices = load_prices(tickers, start=START, end=END)
    assert (prices > 0).all().all(), "Non-positive price in data (e.g. WTI in April 2020)"

    ok = True

    # --- Base result (MA 20/50, 10 bps) ---
    signal = moving_average_signal(prices, 20, 50)
    ok &= check(f"{name}:base", summarise(run_backtest(prices, signal, 10)), expected["base"])

    # --- Cost sensitivity ---
    for bps, key in [(0, "cost_0"), (10, "cost_10"), (30, "cost_30"), (50, "cost_50")]:
        ok &= check(f"{name}:{key}", summarise(run_backtest(prices, signal, bps)), expected[key])

    # --- Parameter sensitivity ---
    for short, long, key in [
        (10, 30, "params_10_30"), (20, 50, "params_20_50"), (20, 80, "params_20_80"),
        (30, 100, "params_30_100"), (40, 120, "params_40_120"), (50, 150, "params_50_150"),
    ]:
        sig = moving_average_signal(prices, short, long)
        ok &= check(f"{name}:{key}", summarise(run_backtest(prices, sig, 10)), expected[key])

    # --- Mean reversion (default parameters) and drawdown breaker ---
    mr_signal = mean_reversion_signal(prices)
    ok &= check(f"{name}:mean_reversion", summarise(run_backtest(prices, mr_signal, 10)), expected["mean_reversion"])
    ok &= check(f"{name}:breaker_off", summarise(run_backtest(prices, mr_signal, 10, limit=None)), expected["breaker_off"])
    ok &= check(f"{name}:breaker_on", summarise(run_backtest(prices, mr_signal, 10, limit=-0.20)), expected["breaker_on"])

    # --- Out-of-sample test: choose on train only, report on test ---
    best = None
    for window, entry in product([10, 20, 40], [-0.5, -1.0, -1.5]):
        sig = mean_reversion_signal(prices, window, entry, 0.0)
        sharpe = run_period(prices, sig, TRAIN)["strategy_sharpe_ratio"]
        if best is None or sharpe > best[0]:
            best = (sharpe, window, entry)
    _, window, entry = best
    print(f"\n  Chosen on {TRAIN[0]} to {TRAIN[1]}: window={window}, entry={entry}\n")
    sig = mean_reversion_signal(prices, window, entry, 0.0)
    ok &= check(f"{name}:oos_train", run_period(prices, sig, TRAIN), expected["oos_train"])
    ok &= check(f"{name}:oos_test", run_period(prices, sig, TEST), expected["oos_test"])

    # --- Lookahead check: same strategies with the execution lag removed ---
    nolag = NOLAG_EXPECTED[name]
    ok &= check(f"{name}:nolag_ma", summarise(run_backtest(prices, signal, 10, lag=0)), nolag["nolag_ma"])
    ok &= check(f"{name}:nolag_mr", summarise(run_backtest(prices, mr_signal, 10, lag=0)), nolag["nolag_mr"])
    return ok


def main():
    all_passed = True
    for name, tickers in UNIVERSES.items():
        all_passed &= verify_universe(name, tickers)

    print()
    if all_passed:
        print("ALL README FIGURES MATCH the current codebase. Safe to publish.")
    else:
        print("SOME FIGURES DO NOT MATCH - update the README or investigate why the code changed.")


if __name__ == "__main__":
    main()