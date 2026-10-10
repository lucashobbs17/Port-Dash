import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import alpha_lab as lab
from signals import load_history

st.set_page_config(page_title="Alpha Lab", layout="wide")
st.title("Alpha Lab")
st.caption("Candidate signals on trial. Nothing here is traded or logged. A signal moves to the Signals page "
           "only if it holds up in the holdout years.")


def colour(v):
    return "" if not isinstance(v, (int, float)) or pd.isna(v) or v == 0 else f"color: {'#1a9850' if v > 0 else '#d73027'}"


c1, c2 = st.columns([1, 4])
cost_bps = c1.number_input("Cost per trade (bps)", 0, 50, 5)
if c2.button("Refresh data"):
    for f in (load_history, lab.load_inventories, lab.load_positioning, lab.load_vol_indices, lab.run_lab):
        f.clear()

try:
    prices, missing = load_history(tuple((p, v[0]) for p, v in lab.PRODUCTS.items()), "2006-01-01")
except Exception as e:
    st.error(f"Price history unavailable ({type(e).__name__}).")
    st.stop()
if prices.shape[1] < 3 or len(prices) < 800:
    st.error("Not enough ETF price history to run the lab.")
    st.stop()

notes = [f"No price data for: {', '.join(missing)}"] if missing else []
eia_key = os.environ.get("EIA_API_KEY")
inventories = pd.DataFrame()
if eia_key:
    try:
        inventories, inv_notes = lab.load_inventories(eia_key)
        notes += inv_notes
    except Exception as e:
        notes.append(f"EIA inventories: failed ({type(e).__name__})")
else:
    notes.append("EIA_API_KEY not set, so the two inventory signals are skipped")
positioning, markets = pd.DataFrame(), {}
try:
    positioning, markets = lab.load_positioning()
    unmatched = [p for p in lab.PRODUCTS if p not in markets]
    if unmatched:
        notes.append("No CFTC positioning for: " + ", ".join(unmatched))
except Exception as e:
    notes.append(f"CFTC positioning: failed ({type(e).__name__})")
for n in notes:
    st.warning(n)

table, groups, products, returns, now = lab.run_lab(prices, cost_bps, inventories, positioning)
st.caption(f"{prices.shape[1]} ETFs, {prices.index[0]:%b %Y} to {prices.index[-1]:%d %b %Y}. "
           f"Holdout starts {pd.Timestamp(lab.HOLDOUT_START):%b %Y}.")

# ---------- ranking ----------
st.subheader("Ranking")
sharpes = ["Sharpe to 2022", "Sharpe holdout", "Sharpe all", "t-stat all", "With risk overlay"]
pct = ["Ann. vol", "Max DD", "Avg exposure", "Turnover/day"]
st.dataframe(table.style.format({**{c: "{:.2f}" for c in sharpes}, **{c: "{:.1%}" for c in pct}, "Years": "{:.1f}"},
                                na_rep="").map(colour, subset=sharpes),
             hide_index=True, height=38 + 35 * len(table))
st.caption("Sorted by the development period (to 2022), which is the only column to use when deciding what to change. "
           "The holdout column is the test: a signal that is positive before 2023 and not after has failed it. "
           "'With risk overlay' is the full-period Sharpe after halving positions in products whose recent "
           "volatility is well above normal. A t-stat under about 2 is not distinguishable from luck.")

# ---------- curves ----------
st.subheader("Growth of 1, each scaled to 10% volatility")
scaled = returns * (0.10 / (returns.std() * np.sqrt(252))).replace([np.inf, -np.inf], np.nan)
curve = (1 + scaled.fillna(0)).cumprod().iloc[::5]
long = curve.rename_axis("date").reset_index().melt("date", var_name="Signal", value_name="value")
lines = alt.Chart(long).mark_line().encode(
    x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=None, scale=alt.Scale(type="log")),
    color=alt.Color("Signal:N", legend=alt.Legend(orient="bottom", columns=3, title=None)),
    tooltip=["date:T", "Signal", alt.Tooltip("value:Q", format=".2f")])
rule = alt.Chart(pd.DataFrame({"date": [pd.Timestamp(lab.HOLDOUT_START)]})).mark_rule(strokeDash=[6, 4], color="grey").encode(x="date:T")
st.altair_chart((lines + rule).properties(height=360))
st.caption("Dashed line: start of the holdout. Scaling uses each signal's full-period volatility, so this chart is "
           "for comparing shapes, not for reading off returns.")

# ---------- breakdowns ----------
left, right = st.columns([2, 3])
with left:
    st.subheader("Sharpe by group")
    st.dataframe(groups.style.format({c: "{:.2f}" for c in groups.columns[1:]}, na_rep="")
                 .map(colour, subset=list(groups.columns[1:])), hide_index=True, height=38 + 35 * len(groups))
with right:
    st.subheader("Sharpe by product")
    st.dataframe(products.style.format({c: "{:.2f}" for c in products.columns[1:]}, na_rep="")
                 .map(colour, subset=list(products.columns[1:])), hide_index=True, height=38 + 35 * len(products))
st.caption("Full period, net of costs. Blank means the signal has no data for that group or product. "
           "Single-product numbers are noisy: with 15 years of data, anything between about -0.5 and +0.5 is noise.")

# ---------- now ----------
st.subheader("Signals now")
signal_cols = [c for c in now.columns if c not in ("Product", "Group", "Vol regime")]
st.dataframe(now.style.format({**{c: "{:+.0f}" for c in signal_cols}, "Vol regime": "{:.2f}x"}, na_rep="")
             .map(colour, subset=signal_cols)
             .map(lambda v: "color: #d73027" if isinstance(v, float) and v > 1.5 else "", subset=["Vol regime"]),
             hide_index=True, height=38 + 35 * len(now))
st.caption("-100 is the strongest short view, +100 the strongest long, blank is no data. Vol-spike reversion is 0 "
           "unless a trade is open. Vol regime is 20-day volatility over 1-year volatility; above 1.5x is flagged "
           "and is where the risk overlay cuts hardest.")

# ---------- risk gauges ----------
st.subheader("Volatility gauges")
try:
    gauges = lab.load_vol_indices()
    st.dataframe(gauges.style.format({"Level": "{:.1f}", "3y percentile": "{:.0%}", "1w change": "{:+.1f}"}, na_rep="unavailable")
                 .map(colour, subset=["1w change"]), hide_index=True)
    st.caption("Option-implied volatility: what the market is pricing for the next month. A high percentile means "
               "the market expects larger moves than it has for most of the last three years.")
except Exception as e:
    st.error(f"Volatility gauges unavailable ({type(e).__name__}).")

if markets:
    with st.expander("CFTC markets matched to each product"):
        st.dataframe(pd.DataFrame({"Product": list(markets), "CFTC market": list(markets.values())}), hide_index=True)

with st.expander("Rules for each signal"):
    st.markdown(f"""
- **Vol-spike reversion**: when a product moves more than {lab.TRIGGER:g}x its normal daily volatility (60 days, measured
  before the move), take the other side at that close. Exit on the first of: {lab.HOLD_DAYS} trading days, a gain of
  {lab.TAKE_PROFIT:.0%}, or another spike in the trade's favour. No stop-loss. Each position targets
  {lab.RISK_PER_POSITION:.0%} annual volatility, capped at {lab.POSITION_CAP:.0%} of capital; total exposure is capped at 100%.
- **Inventory level**: US stocks against the same week in the previous {lab.NORM_YEARS} years. High stocks short, low
  stocks long. Crude stocks drive WTI and Brent, gasoline stocks drive gasoline, gas storage drives Henry Hub.
- **Inventory surprise**: the same, using the weekly change against the normal change for that week.
- **Positioning (fade crowd)**: managed-money net position as a share of open interest, against its own
  {lab.POSITIONING_WINDOW // 52}-year history. Unusually long speculators short, unusually short long.
- **Trend (reference)**: the Signals-page trend rule, for comparison.
- **Timing**: inventories are used from {lab.INVENTORY_LAG_DAYS} days after the week they describe, positioning from
  {lab.POSITIONING_LAG_DAYS} days after the report date, so nothing trades on data before it was public.
- **Sizing**: fundamentals and trend are sized by inverse volatility with absolute weights summing to 1. All positions
  earn the next day's return and pay the cost above on every change.
""")
