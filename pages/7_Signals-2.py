import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from signals import UNIVERSES, load_history, run_research, signals_now
import research_log as rl

st.set_page_config(page_title="Signals", layout="wide")
st.title("Signals")
st.caption("Research only: nothing here places or records a trade.")


def colour(v):
    return "" if pd.isna(v) or v == 0 else f"color: {'#1a9850' if v > 0 else '#d73027'}"


c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
universe_name = c1.selectbox("Universe", list(UNIVERSES))
cost_bps = c2.number_input("Cost per trade (bps)", 0, 50, 5)
start_year = c3.number_input("History from", 2000, 2022, 2005)
if c4.button("Refresh data"):
    load_history.clear()
    run_research.clear()
    rl.fetch_curve.clear()

universe = UNIVERSES[universe_name]
try:
    prices, missing = load_history(tuple(universe["tickers"].items()), f"{start_year}-01-01")
except Exception as e:
    st.error(f"Price history unavailable ({type(e).__name__}).")
    st.stop()
if missing:
    st.warning("No price data for: " + ", ".join(missing))
if prices.shape[1] < 3 or len(prices) < 800:
    st.error("Not enough price history to backtest (need at least 3 products and about 3 years).")
    st.stop()

table, returns, weights, scores, angle, reliability = run_research(prices, universe.get("pairs", {}), cost_bps)
st.caption(f"{prices.shape[1]} products, {prices.index[0]:%b %Y} to {prices.index[-1]:%d %b %Y}. "
           "Yahoo futures prices are front-month contracts joined without roll adjustment, so backtested "
           "returns include jumps on roll days. Treat the numbers as a ranking, not as achievable returns.")

# ---------- 1. ranking ----------
st.subheader("Strategy ranking")
pct = ["Return 3m", "Ann. return", "Ann. vol", "Max DD", "Current DD", "Turnover/day"]
num = ["Sharpe", "t-stat", "Sharpe 3y", "Sharpe 1y"]
shown = table[["Strategy", "Holding up", *num, *pct]]
st.dataframe(
    shown.style.format({**{c: "{:.2f}" for c in num}, **{c: "{:.1%}" for c in pct}}, na_rep="")
    .map(colour, subset=[*num, "Return 3m", "Ann. return"]),
    hide_index=True, height=38 + 35 * len(shown),
    column_config={"Holding up": st.column_config.ProgressColumn("Holding up", min_value=0, max_value=100, format="%.0f")})
st.caption("Net of costs. A t-stat below about 2 means the result is not distinguishable from luck. "
           "Angle picker is the only row that chooses anything from past results, and it is scored only on the "
           "periods after each choice. Long-only basket is the do-nothing benchmark.")

# ---------- 2. curves and correlation ----------
left, right = st.columns([3, 2])
with left:
    st.subheader("Growth of 1")
    curve = (1 + returns).cumprod().iloc[::5]
    long = curve.rename_axis("date").reset_index().melt("date", var_name="Strategy", value_name="value")
    st.altair_chart(alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(type="log")),
        color=alt.Color("Strategy:N", legend=alt.Legend(orient="bottom", columns=3, title=None)),
        tooltip=["date:T", "Strategy", alt.Tooltip("value:Q", format=".2f")]).properties(height=380))
with right:
    st.subheader("Correlation between strategies")
    corr = returns.loc[:, returns.std() > 0].corr()
    st.dataframe(corr.style.format("{:.2f}").map(lambda v: colour(v) if abs(v) >= 0.3 and v < 0.999 else ""),
                 height=38 + 35 * len(corr))
    st.caption("Coloured where the link is 0.3 or stronger. Low or negative numbers are what make a blend worth more than its parts.")

# ---------- 3. signals now ----------
st.subheader("Signals now")
now = signals_now(scores, angle, reliability)
angles = [c for c in now.columns if c not in ("Product", "Picked angle", "View", "Reliability", "Strength")]
st.dataframe(
    now[["Product", "View", "Strength", "Picked angle", "Reliability", *angles]].style
    .format({**{c: "{:+.0f}" for c in angles}, "Reliability": "{:.2f}"}, na_rep="")
    .map(colour, subset=angles)
    .map(lambda v: colour(1 if v == "Long" else -1 if v == "Short" else 0), subset=["View"]),
    hide_index=True, height=38 + 35 * len(now),
    column_config={"Strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.0f")})
st.caption("Angle columns run from -100 (strongest short) to +100 (strongest long); blank means that angle has no "
           "view on the product. Picked angle is whichever had the best 3-year Sharpe on that product up to last "
           "month-end. Strength = conviction of the picked angle x its reliability (that Sharpe, capped at 1).")

# ---------- 4. paper targets ----------
st.subheader("Target positions")
c1, c2 = st.columns([2, 1])
strategy = c1.selectbox("Strategy", table["Strategy"].tolist())
capital = c2.number_input("Paper capital", 1_000, 100_000_000, 100_000, step=10_000)
target = weights[strategy].iloc[-1]
target = target[target.abs() > 0.001].sort_values(ascending=False)
if target.empty:
    st.info("This strategy holds nothing today.")
else:
    st.dataframe(pd.DataFrame({"Product": target.index, "Weight": target.values, "Notional": target.values * capital})
                 .style.format({"Weight": "{:+.1%}", "Notional": "{:+,.0f}"}).map(colour, subset=["Weight", "Notional"]),
                 hide_index=True, height=38 + 35 * len(target))
st.caption("What the strategy would hold at today's close, as a share of capital. Negative is short. "
           "These are targets for a paper book, not orders.")

# ---------- 5. paper track record ----------
st.subheader("Paper track record")
try:
    rl.record_targets(universe_name, weights)
    paper, days = rl.paper_returns(universe_name, prices, cost_bps)
    start = paper.index[0]
    st.caption(f"Targets logged on {days} day(s) since {start:%d %b %Y}. Each visit records that day's targets for "
               "every strategy in this universe; they are held until the next visit, so missed days mean a stale book. "
               "This is the only result on the page with no hindsight in it.")
    if len(paper) < 2:
        st.info("Logging started today. Results appear from the next trading day.")
    else:
        track = pd.DataFrame({
            "Strategy": paper.columns,
            "Paper return": [(1 + paper[c]).prod() - 1 for c in paper],
            "Backtest, same days": [(1 + returns[c].loc[start:].iloc[1:]).prod() - 1 if c in returns else np.nan for c in paper],
        }).sort_values("Paper return", ascending=False)
        st.dataframe(track.style.format({"Paper return": "{:+.2%}", "Backtest, same days": "{:+.2%}"}, na_rep="")
                     .map(colour, subset=["Paper return", "Backtest, same days"]),
                     hide_index=True, height=38 + 35 * len(track))
        st.caption("The two columns should stay close if the page is opened every trading day. "
                   "The paper column includes the cost of entering the book on day one.")
        if len(paper) >= 5:
            long = ((1 + paper).cumprod().rename_axis("date").reset_index()
                    .melt("date", var_name="Strategy", value_name="value"))
            st.altair_chart(alt.Chart(long).mark_line().encode(
                x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("Strategy:N", legend=alt.Legend(orient="bottom", columns=3, title=None)),
            ).properties(height=300))
except Exception as e:
    st.error(f"Paper track record unavailable ({type(e).__name__}).")

# ---------- 6. curve structure ----------
if universe_name == "Commodities":
    st.subheader("Curve structure")
    try:
        curve = rl.fetch_curve()
        days = rl.record_curve(curve)
        curve["State"] = np.select([curve["Carry"] > 0.005, curve["Carry"] < -0.005],
                                   ["Backwardation", "Contango"], "Flat")
        curve.loc[curve["Carry"].isna(), "State"] = "unavailable"
        st.dataframe(
            curve.sort_values("Carry", ascending=False)[["Product", "State", "Carry", "Near", "Near price", "Far", "Far price"]]
            .style.format({"Carry": "{:+.1%}", "Near price": "{:,.2f}", "Far price": "{:,.2f}"}, na_rep="")
            .map(colour, subset=["Carry"]),
            hide_index=True, height=38 + 35 * len(curve))
        st.caption(f"Carry = near price / price of the same contract month a year later, minus 1. Positive "
                   f"(backwardation) means a long position gains as it rolls; negative (contango) means it pays. "
                   f"Recorded on {days} day(s) so far. A carry strategy needs about a year of this before it can "
                   f"be backtested.")
        failed = curve.loc[curve["Carry"].isna(), "Product"].tolist()
        if failed:
            st.warning("No curve data for: " + ", ".join(failed))
    except Exception as e:
        st.error(f"Curve structure unavailable ({type(e).__name__}).")

with st.expander("How the ratings work"):
    st.markdown("""
- **Trend**: each product's return from 12 months ago to 1 month ago, divided by its volatility.
- **Relative momentum**: the same return, ranked across products. Long the top, short the bottom.
- **Spread reversion**: for linked pairs (cracks, Brent-WTI, gold/copper, gold/silver, wheat/corn), how far the ratio
  is from its 1-year average. Bets on it closing.
- **Seasonality**: how the current calendar month has gone in previous years only (needs 5 years).
- **Angle picker**: per product, each month, follows whichever of the four had the best Sharpe on that product over
  the previous 3 years, or takes no view if none was positive.
- **Blend of the four**: equal capital in the four base strategies.
- **Positions**: scores are sized by inverse volatility and scaled so absolute weights sum to 1. Positions decided
  at one close earn the next day's return, and every change in weight is charged the cost above.
- **Holding up (0-100)**: evidence x recent form. Evidence is the t-stat (full marks at 3). Recent form is the 1-year
  Sharpe (1 or more is full marks, 0 is half, -1 or less is zero).
""")
