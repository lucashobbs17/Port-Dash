import pandas as pd

def equal_weight(prices):
    w = prices.notna().astype(float)   
    weight = w.div(w.sum(axis=1), axis=0)    # each row sums to 1
    return weight

def residual_reversal(prices, lookback=5, groups=None):
    daily_returns = prices.pct_change()
    if groups is None:
        residuals = daily_returns.sub(daily_returns.mean(axis=1), axis=0)
    else:
        group_mean = daily_returns.T.groupby(groups).transform('mean').T
        residuals = daily_returns - group_mean
    signal = -residuals.rolling(lookback).mean()
    signal = signal.sub(signal.mean(axis=1), axis=0)
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    return weights

def momentum(prices, lookback=252, skip=21, groups=None):
    past_return = prices.shift(skip) /prices.shift(lookback) - 1
    if groups is None:
        signal = past_return.sub(past_return.mean(axis=1), axis=0)
    else:
        group_mean = past_return.T.groupby(groups).transform('mean').T
        signal = past_return - group_mean
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    return weights