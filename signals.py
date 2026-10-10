"""Research framework: strategies -> backtest -> ranking -> current signal strength.

A strategy is a function  prices -> scores,  one score per product per day in [-1, 1]
(+1 = strongest long view, -1 = strongest short view, NaN = no view), using only data
up to that day. Scores become weights in to_weights(), and run_backtest() adds the
one-day trading lag and the costs. Nothing here knows about positions or orders.
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

UNIVERSES = {
    "Commodities": {
        "tickers": {
            "Brent": "BZ=F", "WTI": "CL=F", "NY Harbor ULSD": "HO=F", "RBOB gasoline": "RB=F",
            "Henry Hub": "NG=F", "Dutch TTF": "TTF=F", "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
            "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F",
        },
        
        "pairs": {
            "Diesel crack": ("NY Harbor ULSD", "WTI"), "Gasoline crack": ("RBOB gasoline", "WTI"),
            "Brent-WTI": ("Brent", "WTI"), "Gold/copper": ("Gold", "Copper"),
            "Gold/silver": ("Gold", "Silver"), "Wheat/corn": ("Wheat", "Corn"),
        },
    },
    "Commodity ETFs": {
        "tickers": {
            "Brent": "BNO", "WTI": "USO", "Gasoline": "UGA", "Henry Hub": "UNG",
            "Gold": "GLD", "Silver": "SLV", "Copper": "CPER",
            "Corn": "CORN", "Wheat": "WEAT", "Soybeans": "SOYB",
        },
        "pairs": {
            "Gasoline crack": ("Gasoline", "WTI"), "Brent-WTI": ("Brent", "WTI"),
            "Gold/copper": ("Gold", "Copper"), "Gold/silver": ("Gold", "Silver"),
            "Wheat/corn": ("Wheat", "Corn"),
        },
    },
}
BASE = ["Trend", "Relative momentum", "Spread reversion", "Seasonality"]


# ---------- data ----------

@st.cache_data(ttl=3600, show_spinner="Pulling price history...")
def load_history(tickers, start="2005-01-01"):
    """tickers: tuple of (name, yahoo_ticker). Returns (prices, list of names with no data)."""
    symbols = [t for _, t in tickers]
    raw = yf.download(symbols, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(symbols[0])
    prices = pd.DataFrame({name: raw[t] for name, t in tickers if t in raw})
    prices = prices.where(prices > 0)                     # WTI printed negative in April 2020
    missing = [name for name, _ in tickers if name not in prices or prices[name].isna().all()]
    prices = prices.drop(columns=[m for m in missing if m in prices]).dropna(how="all")
    return prices.ffill(limit=5), missing


def annual_vol(prices, window=60):
    return prices.pct_change(fill_method=None).rolling(window).std() * np.sqrt(252)


# ---------- strategies: prices -> scores in [-1, 1] ----------

def trend(prices, lookback=252, skip=21):
    """Each product against its own past: return from 12 months ago to 1 month ago, per unit of risk."""
    past = prices.shift(skip) / prices.shift(lookback) - 1
    return (past / annual_vol(prices, 252)).clip(-2, 2) / 2


def relative_momentum(prices, lookback=252, skip=21):
    """Products against each other: long the strongest over the past year, short the weakest."""
    past = prices.shift(skip) / prices.shift(lookback) - 1
    rank, n = past.rank(axis=1), past.notna().sum(axis=1)
    return rank.sub((n + 1) / 2, axis=0).div(((n - 1) / 2).replace(0, np.nan), axis=0)


def spread_reversion(prices, pairs, window=252):
    """Pairs that share a driver: bet that a stretched ratio returns to its 1-year average."""
    scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    used = set()
    for a, b in pairs.values():
        if a in prices and b in prices:
            ratio = np.log(prices[a] / prices[b])
            z = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()
            pos = ((-z).clip(-2, 2) / 2).fillna(0)
            scores[a] += pos
            scores[b] -= pos
            used |= {a, b}
    scores = scores.clip(-1, 1).where(prices.notna())
    scores[[c for c in scores if c not in used]] = np.nan
    return scores


def seasonality(prices, min_years=5):
    """Calendar effect: how this month has gone in previous years only, as a t-statistic."""
    month = prices.index.to_period("M")
    monthly = prices.groupby(month).last().pct_change(fill_method=None)
    by_month = monthly.groupby(monthly.index.month)
    mean = by_month.transform(lambda x: x.expanding(min_years).mean().shift(1))
    std = by_month.transform(lambda x: x.expanding(min_years).std().shift(1))
    count = by_month.transform(lambda x: x.expanding().count().shift(1))
    t = mean / (std / np.sqrt(count))
    return (t.clip(-2, 2) / 2).reindex(month).set_axis(prices.index)


def adaptive(scores, prices, window=756):
    """Angle picker. Each month, per product, follow whichever base strategy had the best
    Sharpe on that product over the previous 3 years; no view if none was positive."""
    ret, vol, month = prices.pct_change(fill_method=None), annual_vol(prices), prices.index.to_period("M")
    names, sharpes = list(scores), []
    for name in names:
        pnl = (scores[name] / vol).shift(1) * ret
        s = pnl.rolling(window).mean() / pnl.rolling(window).std() * np.sqrt(252)
        sharpes.append(s.groupby(month).last().shift(1).reindex(month).set_axis(prices.index))
    stack = np.stack([s.to_numpy(float) for s in sharpes])
    stack = np.where(np.isfinite(stack), stack, -np.inf)
    best, pick = stack.max(axis=0), stack.argmax(axis=0)
    has_view = best > 0
    all_scores = np.stack([scores[n].to_numpy(float) for n in names])
    chosen = np.take_along_axis(all_scores, pick[None], axis=0)[0]
    frame = lambda a: pd.DataFrame(a, index=prices.index, columns=prices.columns)
    score = frame(np.where(has_view, chosen, 0.0)).where(prices.notna())
    angle = frame(np.where(has_view, np.array(names, dtype=object)[pick], "None"))
    return score, angle, frame(np.where(has_view, best, np.nan))


def all_scores(prices, pairs):
    scores = {"Trend": trend(prices), "Relative momentum": relative_momentum(prices),
              "Spread reversion": spread_reversion(prices, pairs), "Seasonality": seasonality(prices)}
    scores["Angle picker"], angle, reliability = adaptive(scores, prices)
    scores["Long-only basket"] = prices.notna().astype(float).where(prices.notna())
    return scores, angle, reliability


# ---------- engine (same conventions as backtest.py) ----------

def to_weights(scores, prices):
    """Size by inverse volatility, then scale so absolute weights sum to 1 (0 when there is no view)."""
    w = scores / annual_vol(prices)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def run_backtest(weights, prices, cost_bps=5):
    weights = weights.fillna(0)
    returns = (prices.pct_change(fill_method=None).fillna(0) * weights.shift(1)).sum(axis=1)
    trades = weights.diff().fillna(weights)
    return returns - cost_bps / 10000 * trades.abs().sum(axis=1)


def sharpe(returns, rf=0.0):
    vol = returns.std() * np.sqrt(252)
    return (returns.mean() * 252 - rf) / vol if len(returns) > 20 and vol > 0 else np.nan


def get_metrics(returns, rf=0.0):
    active = returns.loc[returns.ne(0).idxmax():] if returns.ne(0).any() else returns.iloc[:0]
    if len(active) < 60:
        return None
    curve = (1 + active).cumprod()
    drawdown = curve / curve.cummax() - 1
    years = len(active) / 252
    s = sharpe(active, rf)
    return {"Sharpe": s, "t-stat": s * np.sqrt(years), "Sharpe 3y": sharpe(active.iloc[-756:], rf),
            "Sharpe 1y": sharpe(active.iloc[-252:], rf), "Return 3m": active.iloc[-63:].sum(),
            "Ann. return": active.mean() * 252, "Ann. vol": active.std() * np.sqrt(252),
            "Max DD": drawdown.min(), "Current DD": drawdown.iloc[-1], "Years": years}


def holding_up(m):
    """0-100. Evidence (t-stat, full marks at 3) times recent form (1-year Sharpe: 1 or more is
    full marks, 0 is half, -1 or less is zero)."""
    if m is None or pd.isna(m["t-stat"]) or pd.isna(m["Sharpe 1y"]):
        return np.nan
    return 100 * float(np.clip(m["t-stat"] / 3, 0, 1) * np.clip(0.5 + m["Sharpe 1y"] / 2, 0, 1))


@st.cache_data(ttl=3600, show_spinner="Running backtests...")
def run_research(prices, pairs, cost_bps):
    """Returns (ranking table, daily returns per strategy, weights per strategy, scores, angle, reliability)."""
    scores, angle, reliability = all_scores(prices, pairs)
    weights = {name: to_weights(s, prices) for name, s in scores.items()}
    weights["Blend of the four"] = sum(weights[n] for n in BASE) / len(BASE)
    returns = pd.DataFrame({name: run_backtest(w, prices, cost_bps) for name, w in weights.items()})
    rows = []
    for name, w in weights.items():
        m = get_metrics(returns[name])
        if m is None:
            continue
        rows.append({"Strategy": name, **m, "Turnover/day": w.diff().abs().sum(axis=1).mean(),
                     "Holding up": holding_up(m)})
    table = pd.DataFrame(rows).sort_values("Holding up", ascending=False, na_position="last")
    return table, returns[table["Strategy"].tolist()], weights, scores, angle, reliability


def signals_now(scores, angle, reliability):
    """Latest view per product. Strength = conviction of the picked angle (0-100) times its
    reliability (3-year Sharpe on that product, capped at 1)."""
    last = lambda df: df.ffill(limit=5).iloc[-1]
    out = pd.DataFrame({name: last(scores[name]) * 100 for name in BASE})
    picked = last(scores["Angle picker"]) * 100
    out["Picked angle"] = angle.iloc[-1]
    out["View"] = np.select([picked > 5, picked < -5], ["Long", "Short"], "None")
    out["Reliability"] = last(reliability).clip(0, 1)
    out["Strength"] = (picked.abs() * out["Reliability"]).fillna(0)
    out.loc[out["View"] == "None", "Strength"] = 0
    return out.rename_axis("Product").reset_index().sort_values("Strength", ascending=False)
