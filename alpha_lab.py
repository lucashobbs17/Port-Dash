"""Alpha Lab: candidate signals, tested on commodity ETFs before any of them is promoted to the Signals page.

Returns are always measured on the ETFs (they include roll costs). Signals may use ETF prices,
EIA inventories or CFTC positioning. Parameters are fixed here, on purpose: change them rarely.
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from signals import annual_vol, get_metrics, sharpe, to_weights, trend

HOLDOUT_START = "2023-01-01"          # develop on data before this; judge on data after it
GROUPS = ["Energy", "Metals", "Grains"]
# product: (ETF, group, CFTC contract code, inventory series)
PRODUCTS = {
    "Brent": ("BNO", "Energy", "06765T", "crude"), "WTI": ("USO", "Energy", "067651", "crude"),
    "Gasoline": ("UGA", "Energy", "111659", "gasoline"), "Henry Hub": ("UNG", "Energy", "023651", "natgas"),
    "Gold": ("GLD", "Metals", "088691", None), "Silver": ("SLV", "Metals", "084691", None),
    "Copper": ("CPER", "Metals", "085692", None),
    "Corn": ("CORN", "Grains", "002602", None), "Wheat": ("WEAT", "Grains", "001602", None),
    "Soybeans": ("SOYB", "Grains", "005602", None),
}
EIA_PETROLEUM = {"crude": "WCESTUS1", "gasoline": "WGTSTUS1"}
EIA_GAS = {"natgas": "NW2_EPG0_SWO_R48_BCF"}
VOL_INDICES = {"Oil implied vol (OVX)": "^OVX", "Gold implied vol (GVZ)": "^GVZ", "Equity implied vol (VIX)": "^VIX"}

# vol-spike reversion
TRIGGER, HOLD_DAYS, TAKE_PROFIT, POSITION_CAP, RISK_PER_POSITION = 2.0, 5, 0.05, 0.20, 0.03
# fundamentals
NORM_YEARS, INVENTORY_LAG_DAYS, POSITIONING_WINDOW, POSITIONING_LAG_DAYS = 5, 7, 156, 6


# ---------- data ----------

def _eia(route, series, key):
    r = requests.get(f"https://api.eia.gov/v2/{route}/data/", timeout=60, params={
        "api_key": key, "frequency": "weekly", "data[0]": "value", "facets[series][]": list(series.values()),
        "start": "2000-01-01", "sort[0][column]": "period", "sort[0][direction]": "asc", "length": 5000})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    wide = df.pivot_table(index="period", columns="series", values="value")
    return wide.rename(columns={v: k for k, v in series.items()})


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling EIA inventories...")
def load_inventories(key):
    """Weekly US stocks. Returns (DataFrame indexed by week-ending date, list of notes)."""
    frames, notes = [], []
    for label, route, series in [("petroleum", "petroleum/stoc/wstk", EIA_PETROLEUM),
                                 ("gas storage", "natural-gas/stor/wkly", EIA_GAS)]:
        try:
            frames.append(_eia(route, series, key))
        except Exception as e:
            notes.append(f"EIA {label}: failed ({type(e).__name__})")
    return (pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()), notes


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling CFTC positioning...")
def load_positioning():
    """Managed-money net position as a share of open interest, weekly, per product.
    Returns (DataFrame, {product: market name as reported by the CFTC})."""
    codes = {p: v[2] for p, v in PRODUCTS.items()}
    quoted = ",".join(f"'{c}'" for c in codes.values())
    r = requests.get("https://publicreporting.cftc.gov/resource/72hh-3qpy.json", timeout=90, params={
        "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,market_and_exchange_names,"
                   "open_interest_all,m_money_positions_long_all,m_money_positions_short_all",
        "$where": f"cftc_contract_market_code in ({quoted})", "$limit": 50000})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in ("open_interest_all", "m_money_positions_long_all", "m_money_positions_short_all"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["net"] = (df["m_money_positions_long_all"] - df["m_money_positions_short_all"]) / df["open_interest_all"]
    wide = df.pivot_table(index="date", columns="cftc_contract_market_code", values="net")
    latest = df.sort_values("date").groupby("cftc_contract_market_code")["market_and_exchange_names"].last()
    net = pd.DataFrame({p: wide[c] for p, c in codes.items() if c in wide})
    return net, {p: latest[c] for p, c in codes.items() if c in latest}


@st.cache_data(ttl=3600, show_spinner=False)
def load_vol_indices():
    rows = []
    for name, ticker in VOL_INDICES.items():
        try:
            s = yf.Ticker(ticker).history(period="3y")["Close"].dropna()
            rows.append({"Gauge": name, "Level": s.iloc[-1], "3y percentile": (s < s.iloc[-1]).mean(),
                         "1w change": s.iloc[-1] - s.iloc[-6]})
        except Exception:
            rows.append({"Gauge": name, "Level": np.nan, "3y percentile": np.nan, "1w change": np.nan})
    return pd.DataFrame(rows)


# ---------- signals: each returns scores in [-1, 1], dates x products ----------

def spike_reversion(prices, trigger=TRIGGER, hold=HOLD_DAYS, take_profit=TAKE_PROFIT, vol_window=60):
    """Fade a 1-day move bigger than `trigger` x normal daily vol (measured before the move).
    Exit on the first of: `hold` days, a gain of `take_profit`, or another spike in our favour.
    One position per product at a time; no re-entry on the exit day; no stop-loss."""
    ret = prices.pct_change(fill_method=None)
    z = ret / ret.rolling(vol_window).std().shift(1)
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in prices:
        p, zz, r = prices[c].to_numpy(float), z[c].to_numpy(float), ret[c].to_numpy(float)
        pos, side, entry, held = np.zeros(len(p)), 0.0, np.nan, 0
        for t in range(len(p)):
            spike = abs(zz[t]) > trigger
            if side:
                held += 1
                if held >= hold or side * (p[t] / entry - 1) >= take_profit or (spike and side * r[t] > 0):
                    side = 0.0
            elif spike:
                side, entry, held = -np.sign(r[t]), p[t], 0
            pos[t] = side
        out[c] = pos
    return out.where(prices.notna())


def seasonal_z(s, years=NORM_YEARS):
    """How far each weekly value is from the same ISO week in the previous `years` years, in standard deviations."""
    s = s.dropna()
    iso = s.index.isocalendar()
    df = pd.DataFrame({"v": s.to_numpy(float), "y": iso["year"].to_numpy(), "w": iso["week"].clip(upper=52).to_numpy()})
    table = df.pivot_table(index="y", columns="w", values="v", aggfunc="last")
    prior = table.shift(1)
    z = (table - prior.rolling(years, min_periods=years).mean()) / prior.rolling(years, min_periods=years).std()
    return pd.Series([z.at[y, w] for y, w in zip(df["y"], df["w"])], index=s.index)


def _to_daily(weekly_score, prices, lag_days):
    """Make a weekly series usable only from its publication date, and let it expire after 10 trading days."""
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for product, s in weekly_score.items():
        s = s.dropna()
        s.index = s.index + pd.Timedelta(days=lag_days)
        out[product] = s.reindex(prices.index.union(s.index)).ffill(limit=10).reindex(prices.index)
    return out.where(prices.notna())


def inventory_scores(inventories, prices, surprise):
    """High stocks for the time of year are bearish, low stocks bullish.
    surprise=False uses the level; surprise=True uses the weekly change against the normal change."""
    scores = {}
    for product, (_, _, _, key) in PRODUCTS.items():
        if key in inventories and product in prices:
            series = inventories[key].diff() if surprise else inventories[key]
            scores[product] = -(seasonal_z(series).clip(-2, 2) / 2)
    return _to_daily(scores, prices, INVENTORY_LAG_DAYS)


def positioning_scores(net, prices, window=POSITIONING_WINDOW):
    """Fade the crowd: speculators unusually long against their own 3-year history is bearish."""
    scores = {}
    for product in net:
        if product in prices:
            s = net[product].dropna()
            z = (s - s.rolling(window, min_periods=104).mean()) / s.rolling(window, min_periods=104).std()
            scores[product] = -(z.clip(-2, 2) / 2)
    return _to_daily(scores, prices, POSITIONING_LAG_DAYS)


# ---------- sizing and risk ----------

def capped_weights(scores, prices, risk=RISK_PER_POSITION, cap=POSITION_CAP):
    """Each position targets `risk` of annual volatility, capped at `cap` of capital; total exposure never above 100%."""
    w = (scores * (risk / annual_vol(prices)).clip(upper=cap)).fillna(0)
    return w.div(w.abs().sum(axis=1).clip(lower=1), axis=0)


def risk_overlay(weights, prices, fast=20, slow=252, floor=0.5):
    """Cut a product's weight (down to half) when its recent volatility runs above its 1-year norm."""
    ret = prices.pct_change(fill_method=None)
    return weights * (ret.rolling(slow).std() / ret.rolling(fast).std()).clip(floor, 1).fillna(1)


def vol_regime(prices, fast=20, slow=252):
    ret = prices.pct_change(fill_method=None)
    return (ret.rolling(fast).std() / ret.rolling(slow).std()).iloc[-1]


# ---------- backtest ----------

def contributions(weights, prices, cost_bps=5):
    """Daily P&L per product, net of costs. Its row sum equals signals.run_backtest()."""
    weights = weights.fillna(0)
    gross = prices.pct_change(fill_method=None).fillna(0) * weights.shift(1)
    return gross - cost_bps / 10000 * weights.diff().fillna(weights).abs()


def _active(r):
    return r.loc[r.ne(0).idxmax():] if r.ne(0).any() else r.iloc[:0]


@st.cache_data(ttl=3600, show_spinner="Running lab backtests...")
def run_lab(prices, cost_bps, inventories, positioning):
    scores = {"Vol-spike reversion": spike_reversion(prices)}
    weights = {"Vol-spike reversion": capped_weights(scores["Vol-spike reversion"], prices)}
    extra = {}
    if not inventories.empty:
        extra["Inventory level"] = inventory_scores(inventories, prices, surprise=False)
        extra["Inventory surprise"] = inventory_scores(inventories, prices, surprise=True)
    if not positioning.empty:
        extra["Positioning (fade crowd)"] = positioning_scores(positioning, prices)
    extra["Trend (reference)"] = trend(prices)
    for name, s in extra.items():
        scores[name], weights[name] = s, to_weights(s, prices)

    rows, group_rows, product_rows, returns = [], [], [], {}
    for name, w in weights.items():
        pnl = contributions(w, prices, cost_bps)
        total = pnl.sum(axis=1)
        m = get_metrics(total)
        if m is None:
            continue
        active = _active(total)
        overlay = _active(contributions(risk_overlay(w, prices), prices, cost_bps).sum(axis=1))
        returns[name] = total
        rows.append({"Signal": name, "Sharpe to 2022": sharpe(active.loc[:HOLDOUT_START]),
                     "Sharpe holdout": sharpe(active.loc[HOLDOUT_START:]), "Sharpe all": m["Sharpe"],
                     "t-stat all": m["t-stat"], "With risk overlay": sharpe(overlay),
                     "Ann. vol": m["Ann. vol"], "Max DD": m["Max DD"],
                     "Avg exposure": w.loc[active.index].abs().sum(axis=1).mean(),
                     "Turnover/day": w.loc[active.index].diff().abs().sum(axis=1).mean(), "Years": m["Years"]})
        by_group = {g: sharpe(_active(pnl[[p for p in pnl if PRODUCTS[p][1] == g]].sum(axis=1))) for g in GROUPS}
        group_rows.append({"Signal": name, **by_group, "All": m["Sharpe"]})
        product_rows.append({"Signal": name, **{p: sharpe(_active(pnl[p])) for p in pnl}})
    table = pd.DataFrame(rows).sort_values("Sharpe to 2022", ascending=False, na_position="last")
    order = table["Signal"].tolist()
    by = lambda r: pd.DataFrame(r).set_index("Signal").loc[order].reset_index()
    now = pd.DataFrame({n: scores[n].ffill(limit=5).iloc[-1] * 100 for n in order})
    now.insert(0, "Group", [PRODUCTS[p][1] for p in now.index])
    now["Vol regime"] = vol_regime(prices)
    return table, by(group_rows), by(product_rows), pd.DataFrame(returns)[order], now.rename_axis("Product").reset_index()
