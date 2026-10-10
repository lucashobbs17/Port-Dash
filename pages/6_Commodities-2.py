import os
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import flows
    import research_log
except Exception:          # the page still works without the map and the logs
    flows = research_log = None

st.set_page_config(page_title="Commodities", layout="wide")

COMMODITIES = {
    "Brent": ("BZ=F", "$/bbl"), "WTI": ("CL=F", "$/bbl"),
    "NY Harbor ULSD": ("HO=F", "$/gal"), "RBOB gasoline": ("RB=F", "$/gal"),
    "Henry Hub": ("NG=F", "$/MMBtu"), "Dutch TTF": ("TTF=F", "EUR/MWh"),
    "Gold": ("GC=F", "$/oz"), "Silver": ("SI=F", "$/oz"), "Copper": ("HG=F", "$/lb"),
    "Corn": ("ZC=F", "c/bu"), "Wheat": ("ZW=F", "c/bu"), "Soybeans": ("ZS=F", "c/bu"),
}
MACRO = {
    "Dollar index": ("DX-Y.NYB", "index"), "US 10-year yield": ("^TNX", "%"),
    "VIX": ("^VIX", "index"), "S&P 500": ("^GSPC", "index"),
}
EIA_SERIES = {"Crude ex-SPR": "WCESTUS1", "Distillate": "WDISTUS1", "Gasoline": "WGTSTUS1"}
NEWS_TOPICS = {
    "Oil": ('"crude oil" OR Brent OR OPEC', ["CL=F", "BZ=F"]),
    "Gas": ('"natural gas" OR LNG OR "TTF gas"', ["NG=F", "TTF=F"]),
    "Metals": ('"gold price" OR "copper price" OR "silver price"', ["GC=F", "HG=F"]),
    "Grains": ('corn OR wheat OR soybean futures', ["ZC=F", "ZW=F"]),
}
MONTH_CODES = "FGHJKMNQUVXZ"
PCT = ["1D %", "1W %", "1M %"]
COLS = ["Instrument", "Unit", "Last", *PCT, "52W high", "52W low", "3M trend", "Status"]


# ---------- data ----------

def _history(ticker, period):
    s = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    s.index = s.index.tz_localize(None).normalize()
    return s


@st.cache_data(ttl=900, show_spinner="Pulling prices...")
def load_prices(tickers):
    try:
        raw = yf.download(list(tickers), period="1y", progress=False, auto_adjust=False)["Close"]
    except Exception:
        raw = pd.DataFrame()
    out = {}
    for t in tickers:
        s = raw[t].dropna() if t in raw else pd.Series(dtype=float)
        if s.empty:
            try:
                s = _history(t, "1y")
            except Exception:
                s = pd.Series(dtype=float)
        out[t] = s
    return pd.DataFrame(out), datetime.now()


@st.cache_data(ttl=900, show_spinner=False)
def load_deferred_wti():
    target = pd.Timestamp.today() + pd.DateOffset(months=13)
    for offset in (0, 1, -1):
        d = target + pd.DateOffset(months=offset)
        ticker = f"CL{MONTH_CODES[d.month - 1]}{d:%y}.NYM"
        try:
            s = _history(ticker, "6mo")
            if len(s) > 5:
                return ticker, s
        except Exception:
            pass
    return None, None


def _google_news(query):
    r = requests.get("https://news.google.com/rss/search", timeout=15, headers={"User-Agent": "Mozilla/5.0"},
                     params={"q": f"{query} when:2d", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"})
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        source = it.findtext("source") or ""
        title = (it.findtext("title") or "").removesuffix(f" - {source}")
        when = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate")))
        out.append({"title": title, "url": it.findtext("link"), "when": when.tz_convert("UTC"), "source": source})
    return out


def _yahoo_news(ticker):
    out = []
    for n in yf.Ticker(ticker).news or []:
        c = n.get("content") or n
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link")
        when = c.get("pubDate") or c.get("providerPublishTime")
        when = pd.to_datetime(when, unit="s" if isinstance(when, (int, float)) else None, utc=True)
        source = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
        if c.get("title") and url and not pd.isna(when):
            out.append({"title": c["title"], "url": url, "when": when, "source": source})
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_news():
    news, log, counts = {}, [], {}
    day_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    for topic, (query, tickers) in NEWS_TOPICS.items():
        items = []
        for name, fetch, args in [("Google News", _google_news, [query])] + [(f"Yahoo {t}", _yahoo_news, [t]) for t in tickers]:
            try:
                got = fetch(*args)
                items += got
                log.append(f"{topic}, {name}: {len(got)} headlines")
            except Exception as e:
                log.append(f"{topic}, {name}: failed ({type(e).__name__})")
        seen, unique = set(), []
        for n in sorted(items, key=lambda n: n["when"], reverse=True):
            if n["title"] and n["url"] and n["title"].lower() not in seen:
                seen.add(n["title"].lower())
                unique.append(n)
        news[topic] = unique[:10]
        counts[topic] = sum(n["when"] >= day_ago for n in unique)
    return news, log, counts


@st.cache_data(ttl=900, show_spinner=False)
def load_eia(key):
    r = requests.get("https://api.eia.gov/v2/petroleum/stoc/wstk/data/", timeout=30, params={
        "api_key": key, "frequency": "weekly", "data[0]": "value",
        "facets[series][]": list(EIA_SERIES.values()),
        "start": f"{date.today().year - 6}-01-01",
        "sort[0][column]": "period", "sort[0][direction]": "asc", "length": 5000})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 1000   # thousand bbl -> million bbl
    df["period"] = pd.to_datetime(df["period"])
    return df.pivot_table(index="period", columns="series", values="value")


@st.cache_data(ttl=900, show_spinner=False)
def load_agsi(key):
    rows, page, last = [], 1, 1
    while page <= min(last, 20):
        r = requests.get("https://agsi.gie.eu/api", headers={"x-key": key}, timeout=30, params={
            "type": "eu", "from": f"{date.today().year - 6}-01-01", "size": 300, "page": page})
        r.raise_for_status()
        j = r.json()
        rows += j.get("data", [])
        last = int(j.get("last_page", 1))
        page += 1
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["full"], errors="coerce").values, index=pd.to_datetime(df["gasDayStart"]))
    return s.dropna().sort_index()


# ---------- calculations ----------

def stats(s):
    s = s.dropna()
    if s.empty:
        return None
    last = s.iloc[-1]

    def chg(n):
        if len(s) <= n or s.iloc[-1 - n] == 0:
            return np.nan
        return (last - s.iloc[-1 - n]) / abs(s.iloc[-1 - n])

    age = np.busday_count(s.index[-1].date(), date.today())
    return {"Last": last, "1D %": chg(1), "1W %": chg(5), "1M %": chg(21),
            "52W high": s.max(), "52W low": s.min(), "3M trend": s.iloc[-63:].tolist(),
            "Status": "ok" if age <= 2 else f"stale ({s.index[-1]:%d %b})"}


def stats_frame(series_by_name, units):
    rows = []
    for name, s in series_by_name.items():
        row = stats(s) if s is not None else None
        rows.append({"Instrument": name, "Unit": units[name], **(row or {"Status": "unavailable"})})
    return pd.DataFrame(rows, columns=COLS)


def build_spreads(p):
    cl, ho, rb = p["CL=F"], p["HO=F"], p["RB=F"]
    return {
        "Diesel crack": ho * 42 - cl,
        "Gasoline crack": rb * 42 - cl,
        "3-2-1 crack": (2 * rb * 42 + ho * 42 - 3 * cl) / 3,
        "Brent-WTI": p["BZ=F"] - cl,
        "Gold/copper ratio": p["GC=F"] / p["HG=F"],
    }


# ---------- display ----------

def colour(v):
    return "" if pd.isna(v) or v == 0 else f"color: {'#1a9850' if v > 0 else '#d73027'}"


def show_table(df):
    styled = (df.style
              .format({"Last": "{:,.2f}", "52W high": "{:,.2f}", "52W low": "{:,.2f}",
                       **{c: "{:+.2%}" for c in PCT}}, na_rep="")
              .map(colour, subset=PCT))
    st.dataframe(styled, hide_index=True, height=38 + 35 * len(df),
                 column_config={"3M trend": st.column_config.LineChartColumn("3M trend"),
                                "Status": st.column_config.TextColumn("Status", width="medium")})


def line_chart(s, height=160):
    df = s.rename("value").rename_axis("date").reset_index()
    return alt.Chart(df).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False))).properties(height=height)


def seasonal_chart(s, year, bucket, x_title):
    df = pd.DataFrame({"v": s.values, "year": np.asarray(year), "b": np.asarray(bucket)})
    this = int(df["year"].max())
    hist = (df[df["year"].between(this - 5, this - 1)].groupby("b")["v"]
            .agg(["min", "max", "mean"]).reset_index())
    x = alt.X("b:Q", title=x_title)
    band = alt.Chart(hist).mark_area(opacity=0.25).encode(
        x=x, y=alt.Y("min:Q", title=None, scale=alt.Scale(zero=False)), y2="max:Q")
    avg = alt.Chart(hist).mark_line(strokeDash=[4, 4], color="grey").encode(x=x, y="mean:Q")
    cur = alt.Chart(df[df["year"] == this]).mark_line(color="#d62728", strokeWidth=2.5).encode(x=x, y="v:Q")
    return (band + avg + cur).properties(height=220)


def show_spreads(spreads, curve_ticker):
    curve = "WTI curve (front minus ~12m)"
    names = list(spreads) + ([] if curve in spreads else [curve])
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            if name not in spreads:
                st.metric(name, "skipped")
                st.caption("No dated WTI contract could be fetched, so curve structure is not shown.")
                continue
            s = spreads[name].dropna()
            if len(s) < 2:
                st.metric(name, "unavailable")
                continue
            label = name
            if name.startswith("WTI curve"):
                label = f"WTI curve: {'backwardation' if s.iloc[-1] > 0 else 'contango'} (CL=F minus {curve_ticker})"
            st.metric(label, f"{s.iloc[-1]:,.2f}", f"{s.iloc[-1] - s.iloc[-2]:+.2f} on the day")
            st.altair_chart(line_chart(s.iloc[-126:]))


def show_inventory(name, s, year, bucket, x_title, unit, week_lag):
    s = s.dropna()
    year, bucket = np.asarray(year)[-len(s):], np.asarray(bucket)[-len(s):]
    st.metric(f"{name}, {s.index[-1]:%d %b %Y}", f"{s.iloc[-1]:,.1f}{unit}",
              f"{s.iloc[-1] - s.iloc[-1 - week_lag]:+.1f}{unit} on the week")
    st.altair_chart(seasonal_chart(s, year, bucket, x_title))


def show_inventories():
    eia_key, agsi_key = os.environ.get("EIA_API_KEY"), os.environ.get("AGSI_API_KEY")
    st.caption("Red: this year. Band: range of the previous five years. Dashed: their average.")
    cols = st.columns(4)
    eia = None
    if eia_key:
        try:
            eia = load_eia(eia_key)
        except Exception as e:
            eia = type(e).__name__
    for col, (name, series) in zip(cols, EIA_SERIES.items()):
        with col:
            title = f"US {name} stocks"
            if not eia_key:
                st.metric(title, "key not set")
                st.caption("Set EIA_API_KEY to enable.")
            elif isinstance(eia, str) or series not in eia:
                st.metric(title, "unavailable")
                st.caption(f"EIA request failed ({eia})." if isinstance(eia, str) else "Series missing.")
            else:
                try:
                    s = eia[series].dropna()
                    iso = s.index.isocalendar()
                    show_inventory(title, s, iso["year"], iso["week"], "Week of year", " mb", 1)
                except Exception as e:
                    st.metric(title, "unavailable")
                    st.caption(type(e).__name__)
    with cols[3]:
        title = "EU gas storage"
        if not agsi_key:
            st.metric(title, "key not set")
            st.caption("Set AGSI_API_KEY to enable.")
        else:
            try:
                s = load_agsi(agsi_key)
                show_inventory(title, s, s.index.year, s.index.dayofyear, "Day of year", "%", 7)
            except Exception as e:
                st.metric(title, "unavailable")
                st.caption(f"AGSI+ request failed ({type(e).__name__}).")


def show_news():
    news, log, counts = load_news()
    if research_log and any(counts.values()):
        try:
            research_log.record_headlines(counts)
        except Exception:
            pass
    local = datetime.now().astimezone().tzinfo
    for tab, (topic, items) in zip(st.tabs(list(news)), news.items()):
        with tab:
            if not items:
                st.info("Headlines unavailable.")
            for n in items:
                title = n["title"].replace("[", "(").replace("]", ")")
                st.markdown(f"**{n['when'].tz_convert(local):%a %H:%M}** [{title}]({n['url']}) :gray[{n['source']}]")
    with st.expander("Headline sources"):
        st.text(chr(10).join(log))


def show_calendar():
    today = date.today()
    releases = [("EIA Weekly Petroleum Status Report", 2, "10:30 ET"), ("EIA natural gas storage", 3, "10:30 ET"),
                ("Baker Hughes rig count", 4, "13:00 ET"), ("CFTC Commitments of Traders", 4, "15:30 ET")]
    rows = []
    for name, weekday, when in releases:
        nxt = today + timedelta(days=(weekday - today.weekday()) % 7)
        rows.append({"Release": name, "Next": "today" if nxt == today else f"{nxt:%a %d %b}", "Usual time": when})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Monthly: IEA Oil Market Report, OPEC MOMR, EIA STEO and USDA WASDE all land around mid-month. "
               "Weekly releases slip by a day in weeks with a US federal holiday.")


def show_movers(box, frames):
    allrows = pd.concat(frames, ignore_index=True).dropna(subset=["1D %"])
    top = allrows.loc[allrows["1D %"].abs().sort_values(ascending=False).index].head(3)
    with box:
        st.subheader("Biggest movers today")
        if top.empty:
            st.info("No price data available.")
            return
        for col, (_, r) in zip(st.columns(3), top.iterrows()):
            col.metric(r["Instrument"], f"{r['Last']:,.2f}", f"{r['1D %']:+.2%}")


def section(title, fn, *args):
    st.subheader(title)
    try:
        return fn(*args)
    except Exception as e:
        st.error(f"{title}: unavailable ({type(e).__name__})")


# ---------- page ----------

st.title("Commodities")
if st.button("Refresh data"):
    for f in (load_prices, load_deferred_wti, load_news, load_eia, load_agsi):
        f.clear()
    if flows:
        flows.load_balance.clear()
        flows.load_traffic.clear()

instruments = {**COMMODITIES, **MACRO}
prices, pulled = load_prices(tuple(t for t, _ in instruments.values()))
st.caption(f"Last data pull: {pulled:%a %d %b %Y, %H:%M:%S}. Cached for 15 minutes.")
movers_box, flags_box = st.container(), st.container()


def frame_for(group):
    return stats_frame({n: prices[t] for n, (t, _) in group.items()}, {n: u for n, (_, u) in group.items()})


commodity_df, macro_df = frame_for(COMMODITIES), frame_for(MACRO)
section("Prices", show_table, commodity_df)
section("Headlines", show_news)
if flows:
    section("Flows and chokepoints", flows.show_flows)

try:
    spreads = build_spreads(prices)
    curve_ticker, deferred = load_deferred_wti()
    if curve_ticker:
        spreads["WTI curve (front minus ~12m)"] = prices["CL=F"] - deferred
except Exception:
    spreads, curve_ticker = {}, None
spread_units = {n: "ratio" if "ratio" in n else "$/bbl" for n in spreads}
spread_df = stats_frame(spreads, spread_units)
st.subheader("Spreads")
st.caption("Latest value and 6-month history. Cracks in $/bbl. The curve chart compares the rolling front month "
           "with one fixed dated contract, so the gap between them was longer than 12 months further back.")
try:
    show_spreads(spreads, curve_ticker)
except Exception as e:
    st.error(f"Spreads: unavailable ({type(e).__name__})")

section("Macro", show_table, macro_df)
section("Inventories", show_inventories)
section("Calendar", show_calendar)

try:
    show_movers(movers_box, [commodity_df, macro_df, spread_df])
    flagged = pd.concat([commodity_df, macro_df])
    flagged = flagged[flagged["Status"] != "ok"]
    if not flagged.empty:
        flags_box.warning("Check these: " + ", ".join(f"{r['Instrument']} ({r['Status']})" for _, r in flagged.iterrows()))
except Exception as e:
    movers_box.error(f"Biggest movers: unavailable ({type(e).__name__})")
