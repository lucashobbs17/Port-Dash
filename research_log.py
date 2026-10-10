"""Forward records for the Signals page, kept in research.db (separate from trades.db).

paper_targets: the weights each strategy wanted on each day the page was opened.
curve:         a near and a far futures price per product per day, to build a carry history.
"""
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

from signals import run_backtest

DB = Path(__file__).parent / "research.db"
MONTH_CODES = "FGHJKMNQUVXZ"
ALL, METALS, GRAINS = MONTH_CODES, "HKNUZ", "HKNUZ"
# product: (contract root, exchange suffix, listed delivery months)
CONTRACTS = {
    "Brent": ("BZ", "NYM", ALL), "WTI": ("CL", "NYM", ALL), "NY Harbor ULSD": ("HO", "NYM", ALL),
    "RBOB gasoline": ("RB", "NYM", ALL), "Henry Hub": ("NG", "NYM", ALL),
    "Gold": ("GC", "CMX", "GJMQVZ"), "Silver": ("SI", "CMX", METALS), "Copper": ("HG", "CMX", METALS),
    "Corn": ("ZC", "CBT", GRAINS), "Wheat": ("ZW", "CBT", GRAINS), "Soybeans": ("ZS", "CBT", "FHKNQUX"),
}


def _db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_targets (
        date TEXT, universe TEXT, strategy TEXT, product TEXT, weight REAL,
        PRIMARY KEY (date, universe, strategy, product))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS curve (
        date TEXT, product TEXT, near_ticker TEXT, far_ticker TEXT, near_price REAL, far_price REAL,
        PRIMARY KEY (date, product))""")
    return conn


# ---------- paper book ----------

def record_targets(universe, weights):
    """Save today's target weights for every strategy. Re-opening the page on the same day
    overwrites that day's row; earlier days are never touched."""
    rows = []
    for strategy, w in weights.items():
        day = w.index[-1].strftime("%Y-%m-%d")
        rows += [(day, universe, strategy, product, float(x)) for product, x in w.iloc[-1].fillna(0).items()]
    with _db() as conn:
        conn.executemany("INSERT OR REPLACE INTO paper_targets VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def paper_returns(universe, prices, cost_bps=5):
    """Daily returns of each strategy using only the logged weights, held until the next log."""
    with _db() as conn:
        log = pd.read_sql("SELECT * FROM paper_targets WHERE universe = ?", conn, params=(universe,),
                          parse_dates=["date"])
    if log.empty:
        return pd.DataFrame(), 0
    out = {}
    for strategy, g in log.groupby("strategy"):
        w = g.pivot(index="date", columns="product", values="weight")
        px = prices.loc[w.index[0]:]
        w = w.reindex(px.index).ffill().reindex(columns=px.columns).fillna(0)
        out[strategy] = run_backtest(w, px, cost_bps)
    return pd.DataFrame(out), log["date"].nunique()


# ---------- curve recorder ----------

def curve_contracts(today=None):
    """Near = first listed month at least 2 months out. Far = the same month a year later,
    so seasonal products are compared like for like."""
    today = pd.Timestamp(today or date.today())
    out = {}
    for product, (root, exch, months) in CONTRACTS.items():
        d = today + pd.DateOffset(months=2)
        while MONTH_CODES[d.month - 1] not in months:
            d += pd.DateOffset(months=1)
        code = MONTH_CODES[d.month - 1]
        out[product] = (f"{root}{code}{d:%y}.{exch}", f"{root}{code}{d.year + 1 - 2000:02d}.{exch}")
    return out


@st.cache_data(ttl=3600, show_spinner="Pulling futures curves...")
def fetch_curve():
    contracts = curve_contracts()
    tickers = [t for pair in contracts.values() for t in pair]
    try:
        raw = yf.download(tickers, period="5d", progress=False, auto_adjust=False)["Close"]
    except Exception:
        raw = pd.DataFrame()
    rows = []
    for product, (near, far) in contracts.items():
        last = {t: (raw[t].dropna() if t in raw else pd.Series(dtype=float)) for t in (near, far)}
        ok = all(len(s) for s in last.values())
        rows.append({"Product": product, "Near": near, "Far": far,
                     "Near price": last[near].iloc[-1] if len(last[near]) else None,
                     "Far price": last[far].iloc[-1] if len(last[far]) else None,
                     "date": max(s.index[-1] for s in last.values()).strftime("%Y-%m-%d") if ok else None})
    df = pd.DataFrame(rows)
    df["Carry"] = df["Near price"] / df["Far price"] - 1
    return df


def record_curve(df):
    rows = [(r["date"], r["Product"], r["Near"], r["Far"], float(r["Near price"]), float(r["Far price"]))
            for _, r in df.dropna(subset=["date", "Near price", "Far price"]).iterrows()]
    with _db() as conn:
        conn.executemany("INSERT OR REPLACE INTO curve VALUES (?, ?, ?, ?, ?, ?)", rows)
        days = conn.execute("SELECT COUNT(DISTINCT date) FROM curve").fetchone()[0]
    return days


def curve_history():
    with _db() as conn:
        df = pd.read_sql("SELECT * FROM curve", conn, parse_dates=["date"])
    df["carry"] = df["near_price"] / df["far_price"] - 1
    return df.pivot(index="date", columns="product", values="carry")
