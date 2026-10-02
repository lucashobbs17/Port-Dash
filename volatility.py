import streamlit as st

import yfinance as yf
from positions import get_positions
import pandas as pd
@st.cache_data(ttl=300)
def get_price_history(period='2y', base='USD'):
    tickers = get_positions()['ticker'].tolist()
    prices = yf.download(tickers, period=period)['Close'].ffill()

    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        st.warning(f"No price data for: {', '.join(missing)}. Check the ticker symbol.")
        prices = prices.drop(columns=missing)

    if base is None:
        return prices
    for ticker, ccy in get_currencies().items():
        if ccy != base and ticker in prices.columns:
            fx = yf.download(f'{ccy}{base}=X', period=period)['Close'].squeeze()
            prices[ticker] = prices[ticker] * fx.reindex(prices.index).ffill()
    return prices

def get_daily_returns():
    prices = get_price_history()
    returns = prices.pct_change().dropna()
    return returns

def get_rolling_volatility(window=20):
    returns = get_daily_returns()
    rolling_vol = returns.rolling(window).std() * (252 ** 0.5)
    return rolling_vol
@st.cache_data(ttl=86400)

def get_beta():
    tickers = get_positions()['ticker'].tolist()
    betas = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        betas[ticker] = stock.info.get('beta')
    return betas

def get_sharpe_ratios(rf=0.04):
    returns = get_daily_returns()
    mean_ret = returns.mean() * 252
    vol = returns.std() * (252 ** 0.5)
    sharpe = (mean_ret - rf) / vol
    return sharpe

def get_vol_risk_summary():
    latest_vol = get_rolling_volatility(20).iloc[-1]

    sharpe = get_sharpe_ratios()
    beta = get_beta()

    summary = pd.DataFrame({
        'volatility': latest_vol,

        'sharpe': sharpe,
        'beta': pd.Series(beta)
    })
    summary.index.name = 'ticker'
    return summary.reset_index()
@st.cache_data(ttl=86400)
def get_currencies():
    tickers = get_positions()['ticker'].tolist()
    return {t: yf.Ticker(t).fast_info['currency'] for t in tickers}

@st.cache_data(ttl=300)
def get_price_history(period='2y', base='USD'):
    tickers = get_positions()['ticker'].tolist()
    prices = yf.download(tickers, period=period)['Close'].ffill()
    if base is None:
        return prices
    for ticker, ccy in get_currencies().items():
        if ccy != base:
            fx = yf.download(f'{ccy}{base}=X', period=period)['Close'].squeeze()
            prices[ticker] = prices[ticker] * fx.reindex(prices.index).ffill()
    return prices

if __name__ == '__main__':
    price_history = get_price_history()
    daily_returns = get_daily_returns()
    rolling_volatility = get_rolling_volatility(20)

    beta_values = get_beta()
    sharpe_ratios = get_sharpe_ratios()

    vol_risk_summary = get_vol_risk_summary()

