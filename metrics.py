import pandas as pd
import math

# 252 is the approx. num of trading days in a year
ANNUAL_TRADING_DAYS = 252

def total_return(equity_curve):
    # To get how much the money grew/diminished over the entire period
    starting_val = equity_curve.iloc[0]
    ending_val = equity_curve.iloc[-1]
    return (ending_val/starting_val)-1

def annualised_return(equity_curve):
    # Scaling the total return up/down to a "per year" rate so different time periods are comparable
    annual_ret = (1 + total_return(equity_curve=equity_curve)) ** (ANNUAL_TRADING_DAYS/ len(equity_curve)) - 1
    return annual_ret

def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    avg_excess_returns = (daily_returns - (risk_free_rate / ANNUAL_TRADING_DAYS)).mean()
    daily_returns_std = daily_returns.std()
    ratio = avg_excess_returns / daily_returns_std
    annualised_ratio = ratio * math.sqrt(ANNUAL_TRADING_DAYS)
    return annualised_ratio

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1
    max_drawdown_value = drawdowns.min()
    return max_drawdown_value

def summarise(result):
    summary = {}
    summary["strategy_total_return"] = total_return(result["strategy_equity"])
    summary["strategy_annualised_return"] = annualised_return(result["strategy_equity"])
    summary["strategy_sharpe_ratio"] = sharpe_ratio(result["strategy_return"])
    summary["strategy_max_drawdown"] = max_drawdown(result["strategy_equity"])
    summary["benchmark_total_return"] = total_return(result["benchmark_equity"])
    summary["benchmark_annualised_return"] = annualised_return(result["benchmark_equity"])
    summary["benchmark_sharpe_ratio"] = sharpe_ratio(result["benchmark_return"])
    summary["benchmark_max_drawdown"] = max_drawdown(result["benchmark_equity"])
    return summary