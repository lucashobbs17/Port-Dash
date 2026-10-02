import altair as alt
import pandas as pd
import streamlit as st
from positions import get_trades, get_full_positions
from volatility import get_price_history, get_vol_risk_summary, get_currencies
from risk import get_outright_var, get_position_weights
from movements import get_daily_moves, get_since_purchase

def fmt(x, spec):
    return "n/a" if x is None or pd.isna(x) else format(x, spec)

st.title("Stock Detail")

ticker = st.selectbox("Ticker", sorted(get_price_history().columns))
ccy = get_currencies().get(ticker, "USD")

pos = get_full_positions().set_index('ticker').loc[ticker]
risk = get_vol_risk_summary().set_index('ticker').loc[ticker]
move = get_daily_moves().set_index('ticker').loc[ticker]
since = get_since_purchase().loc[ticker]
weight = get_position_weights().set_index('ticker').loc[ticker]
var = get_outright_var()[ticker]

st.subheader("Position")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Quantity", fmt(pos['signed_quantity'], ',.0f'))
c2.metric(f"Avg price ({ccy})", fmt(pos['avg_price'], ',.2f'))
c3.metric(f"Unrealized P&L ({ccy})", fmt(pos['unrealized_pnl'], ',.2f'))
c4.metric(f"Realized P&L ({ccy})", fmt(pos['realized_pnl'], ',.2f'))

st.subheader("Performance & risk")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Return since purchase", fmt(since['return_since_purchase'], '.2%'))
c2.metric("Drawdown since purchase", fmt(since['drawdown_since_purchase'], '.2%'))
c3.metric("Today's move", fmt(move['return'], '.2%'), f"z = {fmt(move['z_score'], '.2f')}", delta_color="off")
c4.metric("Portfolio weight", fmt(weight['weight'], '.1%'))

c1, c2, c3, c4 = st.columns(4)
c1.metric("20d volatility", fmt(risk['volatility'], '.1%'))
c2.metric("Sharpe", fmt(risk['sharpe'], '.2f'))
c3.metric("Beta", fmt(risk['beta'], '.2f'))
c4.metric("1-day VaR (95%, USD)", f"${fmt(var, ',.2f')}")

st.subheader(f"Price ({ccy}) with your trades")
prices = get_price_history(base=None)[ticker].reset_index()
prices.columns = ['date', 'price']

trades = get_trades()
trades = trades[trades['ticker'] == ticker].copy()
trades['date'] = pd.to_datetime(trades['date'])

line = alt.Chart(prices).mark_line().encode(
    x='date:T', y=alt.Y('price:Q', scale=alt.Scale(zero=False)))
points = alt.Chart(trades).mark_point(size=120, filled=True).encode(
    x='date:T', y='price:Q',
    color=alt.Color('side:N', scale=alt.Scale(domain=['buy', 'sell'], range=['green', 'red'])),
    tooltip=['date:T', 'side', 'quantity', 'price'])
st.altair_chart((line + points).interactive())

st.subheader("Trades")
st.dataframe(trades[['trade_id', 'date', 'side', 'quantity', 'price']], hide_index=True)