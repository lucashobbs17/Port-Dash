# Portfolio Risk Dashboard

A local, auto-refreshing risk and P&L dashboard for an equity portfolio, built with Python and Streamlit. Trades are recorded in a SQLite log; positions, P&L, volatility, VaR and market moves are all derived from that log plus market data from yfinance.

## Features

**Positions Overview**: current holdings with signed quantity, average cost, unrealized and realized P&L (realized P&L tracked trade by trade using a running average cost).

**Volatility & Risk**: per-stock annualised volatility (20-day and 60-day rolling), Sharpe ratio (risk-free rate 4%) and beta.

**Portfolio Risk**: position weights, return correlation matrix, outright (standalone) VaR per position, portfolio VaR, the diversification benefit between them, and weighted portfolio beta.

**Movements & Monitoring**: today's move per stock in % and dollar terms, a z-score against each stock's own recent volatility (moves beyond ±2σ are flagged), drawdown from the 2-year high, and return and drawdown since purchase. Refreshes automatically every 5 minutes.

**Stock Detail**: a per-ticker view combining position, performance and risk metrics with a price chart marking every buy and sell at its fill price.

## Risk methodology

- **VaR** is historical, 1-day, 95% confidence, using a 2-year lookback. Portfolio VaR is the 5th percentile of the weighted daily portfolio return series, so correlation between holdings is captured directly from realised co-movement rather than an assumed distribution.
- **Diversification benefit** = sum of outright VaRs − portfolio VaR. It measures how much risk is offset by positions not moving together; near zero means the portfolio is behaving like one concentrated bet.
- **Z-scores** divide today's return by the standard deviation of the previous 60 days, excluding today so the move being tested does not inflate its own baseline.
- **Volatility** is annualised with √252. The 20-day window is used for monitoring (reactive); the 60-day window is smoother and better suited to sizing.

## Multi-currency handling

Each ticker's quote currency is detected automatically. For portfolio-level calculations (weights, VaR, dollar moves), price histories are converted to USD with daily FX rates before returns are computed, so VaR includes currency risk. Position-level P&L and since-purchase performance stay in each stock's local currency, matching the price the trade was recorded at. Prices are forward-filled across differing exchange holidays.

## Design notes

- **Append-only trade log** with soft deletes, so the trade history is never lost.
- **One batched, cached price download** (`st.cache_data`, 5-minute TTL) feeds every page, instead of per-ticker requests.
- Bad or delisted tickers are dropped with a warning rather than breaking the dashboard.

## Project structure

```
app.py                  Streamlit entry point
positions.py            Trade log access, positions, realized/unrealized P&L
volatility.py           Price history, FX conversion, returns, vol, Sharpe, beta
risk.py                 Weights, correlation, VaR, portfolio beta
movements.py            Daily moves, z-scores, drawdowns, since-purchase stats
manage_trades.py        Add, soft-delete and correct trades
pages/                  One file per dashboard page
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Add trades through `manage_trades.py`:

```python
add_trade("2026-09-15", "AAPL", "buy", 10, 230.50)
```

Non-US listings need their exchange suffix (e.g. `BAS.DE` for Xetra, `AYV.PA` for Euronext Paris).

## Limitations

- Historical VaR assumes the next day resembles the past two years; it understates risk going into regime changes.
- Stocks with less than two years of history have noisier risk estimates.
- yfinance data is delayed (~15 minutes) and unofficial, so the dashboard is for monitoring, not execution.
- London-listed stocks quoted in pence (GBp) are not yet handled.
