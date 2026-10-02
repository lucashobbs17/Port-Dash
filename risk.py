import yfinance as yf
from positions import get_positions
import pandas as pd
from volatility import get_beta, get_price_history, get_daily_returns

def get_position_weights():
    positions = get_positions()
    latest_prices = get_price_history().iloc[-1]
    value = positions['signed_quantity'] * positions['ticker'].map(latest_prices)
    total_value = value.sum()
    positions['value'] = value
    positions['weight'] = value / total_value
    return positions[['ticker', 'value', 'weight']]

def get_portfolio_beta():
    weights = get_position_weights().set_index('ticker')['weight']
    betas = pd.Series(get_beta())
    portfolio_beta = (weights * betas).sum()
    return portfolio_beta

def get_correlation_matrix():
    returns = get_daily_returns().dropna()
    correlation_matrix = returns.corr()
    return correlation_matrix

def get_outright_var(confidence_level=0.05):
    returns = get_daily_returns().dropna()
    qv = returns.quantile(confidence_level)
    var = -qv * get_position_weights().set_index('ticker')['value']
    return var

def get_portfolio_var(confidence_level=0.05):
    returns = get_daily_returns().dropna()
    weights = get_position_weights().set_index('ticker')['weight']
    portfolio_returns = (returns * weights).sum(axis=1)
    total_value = get_position_weights()['value'].sum()
    return -portfolio_returns.quantile(confidence_level) * total_value

if __name__ == '__main__':
    position_weights = get_position_weights()
    print(position_weights)
    print(len(get_daily_returns().dropna()))
    var = get_outright_var()
    print(var)
    correlation_matrix = get_correlation_matrix()
    portfolio_var = get_portfolio_var()
    print("Portfolio VaR:", portfolio_var)
    print("Difference:", get_outright_var().sum() - get_portfolio_var())
    portfolio_beta = get_portfolio_beta()
    print("Portfolio Beta:", portfolio_beta)

from volatility import get_price_history
p = get_price_history()
print(p.tail())
print(p.isna().sum())