"""Flow map: who produces and consumes oil and gas, the sea lanes between them, and the chokepoints on the way.

Sources
- Country production and consumption: Energy Institute Statistical Review, via Our World in Data (annual, TWh).
- Chokepoint volumes: EIA, World Oil Transit Chokepoints (updated 3 March 2026). Typed in below.
- Live chokepoint traffic: IMF PortWatch daily transit counts.
Sea lanes are schematic: they show the route, not the volume.
"""
from io import StringIO

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

OWID = "https://ourworldindata.org/grapher/{name}.csv?csvType=full&useColumnShortNames=true"
PORTWATCH = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
             "Daily_Chokepoints_Data/FeatureServer/0/query")
TWH_PER_MBD, TWH_PER_BCM = 579.0, 10.0      # rough conversions, for display only

# EIA Table 1 (million b/d of crude and petroleum liquids) and LNG table (billion cubic feet/day).
# name: (lat, lon, PortWatch id, oil 2023, oil 2024, oil 1H25, LNG 2023, LNG 2024, LNG 1H25)
CHOKEPOINTS = {
    "Strait of Malacca": (2.5, 101.3, "chokepoint5", 24.0, 22.5, 23.2, 9.4, 10.1, 9.2),
    "Strait of Hormuz": (26.6, 56.4, "chokepoint6", 21.8, 20.7, 20.9, 10.6, 10.5, 11.4),
    "Suez Canal and SUMED": (30.5, 32.4, "chokepoint1", 8.8, 4.8, 4.9, 4.1, 0.5, 0.9),
    "Bab el-Mandeb": (12.6, 43.4, "chokepoint4", 9.3, 4.1, 4.2, 4.2, 0.0, 0.0),
    "Danish Straits": (55.7, 12.7, "chokepoint10", 5.0, 4.9, 4.9, 1.4, 1.4, 1.6),
    "Turkish Straits": (41.1, 29.1, "chokepoint3", 3.5, 3.6, 3.7, 0.5, 0.5, 0.6),
    "Panama Canal": (9.1, -79.7, "chokepoint2", 2.2, 2.0, 2.3, None, None, None),
    "Cape of Good Hope": (-34.4, 18.5, "chokepoint7", 6.2, 9.3, 9.1, 2.1, 7.8, 5.7),
}
WORLD_MARITIME_OIL_1H25 = 79.8

# Schematic sea lanes as [lon, lat] waypoints.
LANES = {
    "Gulf to East Asia": [[50.5, 27], [56.4, 26.6], [60, 23], [72, 12], [80, 5.5], [95, 6], [101.3, 2.5], [104.5, 1.2],
                          [110, 8], [115, 18], [121, 25], [123, 31]],
    "Gulf to Europe via Suez": [[56.4, 26.6], [60, 22], [55, 14], [45, 12], [43.4, 12.6], [39, 20], [33.5, 28],
                                [32.4, 30.5], [32.3, 31.6], [20, 34.5], [12, 37.5], [5, 38], [-5.5, 36], [-10, 40],
                                [-6, 48], [2, 51]],
    "Gulf to Atlantic via the Cape": [[56.4, 26.6], [60, 20], [52, 5], [42, -15], [35, -28], [18.5, -35.5], [8, -20],
                                      [-5, 0], [-18, 20], [-12, 40]],
    "West Africa to East Asia": [[5, 3], [8, -15], [18.5, -35.5], [40, -32], [70, -15], [95, 2], [101.3, 2.5]],
    "US Gulf to Europe": [[-93, 28.5], [-85, 24.5], [-79.5, 26], [-65, 34], [-35, 45], [-8, 49], [2, 51]],
    "US Gulf to Asia via Panama": [[-93, 28.5], [-86, 22], [-80.5, 11], [-79.7, 9.1], [-79.5, 7.5], [-100, 10],
                                   [-140, 22], [-179.9, 30]],
    "Pacific to East Asia": [[179.9, 30], [150, 33], [140, 34]],
    "Baltic to North Sea": [[28.5, 60], [20, 58], [14, 55], [12.7, 55.7], [11, 57.5], [6, 57], [3, 52.5]],
    "Black Sea to Mediterranean": [[37.8, 44.6], [32, 43], [29.1, 41.1], [26.5, 40.2], [25, 37], [20, 34.5]],
    "Australia to East Asia": [[116, -20], [118, -10], [121, 5], [125, 20], [133, 31], [138, 34]],
    "Qatar LNG to South Asia": [[51.6, 25.9], [56.4, 26.6], [62, 23], [68, 21]],
}

# Country centroids (lat, lon), close enough for a bubble.
CENTROIDS = {
    "USA": (39, -98), "CAN": (57, -105), "MEX": (23.5, -102), "BRA": (-10, -52), "ARG": (-35, -65), "COL": (4, -73),
    "VEN": (7, -66), "ECU": (-1.5, -78), "GUY": (5, -59), "TTO": (10.5, -61.3), "PER": (-10, -76), "CHL": (-33, -71),
    "NOR": (61, 9), "GBR": (54, -2), "DEU": (51, 10), "FRA": (46.5, 2.5), "ITA": (42.5, 12.5), "ESP": (40, -4),
    "NLD": (52.2, 5.5), "BEL": (50.6, 4.6), "POL": (52, 19.5), "TUR": (39, 35), "RUS": (61, 90), "KAZ": (48, 67),
    "AZE": (40.3, 47.7), "TKM": (39, 59.5), "UZB": (41.5, 64), "UKR": (49, 31.5), "SAU": (24, 45), "IRN": (32.5, 54),
    "IRQ": (33, 43.5), "KWT": (29.3, 47.6), "ARE": (24, 54), "QAT": (25.3, 51.2), "OMN": (21, 57), "EGY": (26.5, 30),
    "DZA": (28, 2.5), "LBY": (27, 17.5), "NGA": (9.5, 8), "AGO": (-12.5, 17.5), "ZAF": (-29, 24.5), "CHN": (35, 103),
    "IND": (22, 79), "JPN": (36.5, 138.5), "KOR": (36.3, 127.9), "TWN": (23.7, 121), "IDN": (-2, 117), "MYS": (4, 102),
    "THA": (15.5, 101), "VNM": (16, 107.5), "SGP": (1.35, 103.8), "PHL": (13, 122), "PAK": (30, 69.5), "BGD": (24, 90),
    "AUS": (-25, 134), "NZL": (-42, 173), "ISR": (31.3, 34.9), "GRC": (39, 22), "SWE": (62, 15.5), "ROU": (46, 25),
    "BHR": (26.05, 50.55), "BRN": (4.5, 114.7), "COG": (-1, 15.5), "GAB": (-0.7, 11.7), "TCD": (15, 19),
    "SSD": (7.5, 30), "GNQ": (1.6, 10.5), "GHA": (8, -1.2), "HUN": (47, 19.5), "CZE": (49.8, 15.5), "AUT": (47.5, 14),
    "CHE": (46.8, 8.2), "PRT": (39.6, -8), "FIN": (64, 26), "DNK": (56, 9.5), "IRL": (53.2, -8), "BLR": (53.5, 28),
    "HKG": (22.3, 114.2), "MAR": (32, -6.5), "LKA": (7.7, 80.7), "TUN": (34, 9.5), "YEM": (15.5, 47.5),
    "SYR": (35, 38.5), "PNG": (-6.5, 145), "MOZ": (-18, 35.5), "TZA": (-6.5, 35), "BGR": (42.7, 25.3),
    "SVK": (48.7, 19.5), "HRV": (45.2, 15.5), "LTU": (55.3, 24), "SRB": (44, 21), "CUB": (21.7, -79),
    "BOL": (-17, -65), "MMR": (21, 96.5), "KHM": (12.7, 105), "ETH": (9, 39.5), "KEN": (0.5, 38), "SDN": (15.5, 30.5),
    "CIV": (7.6, -5.5), "CMR": (5.7, 12.7), "SEN": (14.5, -14.5), "LUX": (49.8, 6.1), "SVN": (46.1, 14.8),
    "EST": (58.7, 25.5), "LVA": (57, 25), "ISL": (65, -18.5), "CYP": (35, 33.2), "JOR": (31.2, 36.5), "LBN": (33.9, 35.9),
}


@st.cache_data(ttl=24 * 3600, show_spinner="Pulling country production and consumption...")
def load_balance(fuel):
    """fuel: 'oil' or 'gas'. Latest-year production, consumption and net balance per country, in TWh."""
    frames = {}
    for kind in ("production", "consumption"):
        r = requests.get(OWID.format(name=f"{fuel}-{kind}-by-country"), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["entity", "code", "year", "value"]
        frames[kind] = df[df["code"].notna() & ~df["code"].astype(str).str.startswith("OWID")]
    year = int(min(f["year"].max() for f in frames.values()))
    pick = lambda k: frames[k][frames[k]["year"] == year].set_index("code")
    prod, cons = pick("production"), pick("consumption")
    out = pd.DataFrame({"Country": prod["entity"].combine_first(cons["entity"]),
                        "Production": prod["value"], "Consumption": cons["value"]}).fillna({"Production": 0, "Consumption": 0})
    out["Net"] = out["Production"] - out["Consumption"]
    return out.rename_axis("code").reset_index(), year


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling chokepoint traffic...")
def load_traffic():
    """Tanker transits per chokepoint: last 7 days against the previous year. Returns (DataFrame, notes)."""
    rows, notes = [], []
    for name, (_, _, pid, *_rest) in CHOKEPOINTS.items():
        try:
            r = requests.get(PORTWATCH, timeout=30, params={
                "where": f"portid='{pid}'", "outFields": "date,n_tanker", "orderByFields": "date DESC",
                "resultRecordCount": 380, "f": "json"})
            r.raise_for_status()
            df = pd.DataFrame([f["attributes"] for f in r.json()["features"]])
            numeric = pd.api.types.is_numeric_dtype(df["date"])
            df["date"] = pd.to_datetime(df["date"], unit="ms" if numeric else None)
            s = df.set_index("date")["n_tanker"].astype(float).sort_index()
            rows.append({"Chokepoint": name, "Tankers/day, last 7d": s.iloc[-7:].mean(),
                         "Tankers/day, prior year": s.iloc[:-7].mean(), "As of": s.index[-1]})
        except Exception as e:
            notes.append(f"{name}: {type(e).__name__}")
    df = pd.DataFrame(rows, columns=["Chokepoint", "Tankers/day, last 7d", "Tankers/day, prior year", "As of"])
    df["vs normal"] = df["Tankers/day, last 7d"] / df["Tankers/day, prior year"] - 1
    return df, notes


def chokepoint_table(fuel, traffic):
    i = 3 if fuel == "oil" else 6
    unit = "mb/d" if fuel == "oil" else "bcf/d"
    df = pd.DataFrame([{"Chokepoint": n, "lat": v[0], "lon": v[1], f"2023 ({unit})": v[i], f"2024 ({unit})": v[i + 1],
                        f"1H25 ({unit})": v[i + 2]} for n, v in CHOKEPOINTS.items()])
    if fuel == "oil":
        df["Share of seaborne oil"] = df[f"1H25 ({unit})"] / WORLD_MARITIME_OIL_1H25
    return df.merge(traffic, on="Chokepoint", how="left") if not traffic.empty else df


def build_map(balance, chokepoints, fuel):
    unit = "mb/d" if fuel == "oil" else "bcf/d"
    per, label = (TWH_PER_MBD, "mb/d") if fuel == "oil" else (TWH_PER_BCM, "bcm/yr")
    b = balance[balance["code"].isin(CENTROIDS)].copy()
    b["lat"] = b["code"].map(lambda c: CENTROIDS[c][0])
    b["lon"] = b["code"].map(lambda c: CENTROIDS[c][1])
    b = b[b["Net"].abs() > b["Net"].abs().max() * 0.004]
    b["radius"] = np.sqrt(b["Net"].abs() / b["Net"].abs().max()) * 900_000
    b["colour"] = b["Net"].map(lambda v: [26, 152, 80, 150] if v > 0 else [215, 48, 39, 150])
    b["tip"] = b.apply(lambda r: f"<b>{r['Country']}</b><br/>{'Net exporter' if r['Net'] > 0 else 'Net importer'}: "
                                 f"about {abs(r['Net']) / per:,.1f} {label}<br/>Produces {r['Production'] / per:,.1f}, "
                                 f"uses {r['Consumption'] / per:,.1f}", axis=1)
    c = chokepoints.copy()
    col = f"1H25 ({unit})"
    c["radius"] = np.sqrt(c[col].fillna(0) / c[col].max()) * 330_000 + 60_000

    def status(r):
        v = r.get("vs normal")
        if v is None or pd.isna(v):
            return [90, 90, 90, 230]
        return [200, 30, 30, 240] if v < -0.3 else [240, 150, 0, 240] if v < -0.1 else [40, 90, 200, 230]

    c["colour"] = c.apply(status, axis=1)
    live = lambda r: "" if pd.isna(r.get("vs normal", np.nan)) else f"<br/>Tanker transits {r['vs normal']:+.0%} vs the past year"
    c["tip"] = c.apply(lambda r: f"<b>{r['Chokepoint']}</b><br/>"
                                 + (f"{r[col]:.1f} {unit} (1H 2025)" if pd.notna(r[col]) else "No EIA figure")
                                 + live(r), axis=1)
    lanes = pd.DataFrame([{"path": p, "tip": f"<b>{n}</b><br/>Schematic sea lane"} for n, p in LANES.items()])
    layers = [
        pdk.Layer("PathLayer", lanes, get_path="path", get_color=[120, 120, 120, 150], width_min_pixels=1.5,
                  get_width=20000, pickable=True),
        pdk.Layer("ScatterplotLayer", b, get_position="[lon, lat]", get_radius="radius", get_fill_color="colour",
                  pickable=True, radius_min_pixels=2),
        pdk.Layer("ScatterplotLayer", c, get_position="[lon, lat]", get_radius="radius", get_fill_color="colour",
                  get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2, pickable=True, radius_min_pixels=7),
        pdk.Layer("TextLayer", c, get_position="[lon, lat]", get_text="Chokepoint", get_size=12,
                  get_color=[90, 90, 90], get_pixel_offset=[0, -18], get_alignment_baseline="'bottom'"),
    ]
    return pdk.Deck(layers=layers, map_style=None, tooltip={"html": "{tip}"},
                    initial_view_state=pdk.ViewState(latitude=22, longitude=45, zoom=1.3, min_zoom=0.8))


def show_flows():
    fuel = st.radio("Commodity", ["oil", "gas"], horizontal=True, format_func=lambda f: "Oil" if f == "oil" else "Natural gas and LNG")
    per, label = (TWH_PER_MBD, "mb/d") if fuel == "oil" else (TWH_PER_BCM, "bcm/yr")
    try:
        traffic, notes = load_traffic()
    except Exception as e:
        traffic, notes = pd.DataFrame(), [type(e).__name__]
    chokepoints = chokepoint_table(fuel, traffic)
    try:
        balance, year = load_balance(fuel)
    except Exception as e:
        balance, year = pd.DataFrame(columns=["code", "Country", "Production", "Consumption", "Net"]), None
        st.warning(f"Country production and consumption unavailable ({type(e).__name__}); showing chokepoints only.")

    st.pydeck_chart(build_map(balance, chokepoints, fuel), height=520)
    st.caption("Green circles are net exporters, red are net importers, sized by the gap between what the country "
               "produces and uses" + (f" ({year})" if year else "") + ". Ringed circles are chokepoints, sized by "
               "EIA volume: blue is normal tanker traffic, amber is 10% or more below the past year, red is 30% or "
               "more below, grey is no live data. Grey lines are schematic sea lanes, not volumes. Hover for numbers.")

    st.markdown("**Chokepoints**")
    table = chokepoints.drop(columns=["lat", "lon"])
    fmt = {c: "{:.1f}" for c in table.columns if "(" in c}
    fmt.update({"Share of seaborne oil": "{:.0%}", "Tankers/day, last 7d": "{:.0f}", "Tankers/day, prior year": "{:.0f}",
                "vs normal": "{:+.0%}", "As of": "{:%d %b}"})
    styled = table.style.format({k: v for k, v in fmt.items() if k in table}, na_rep="")
    if "vs normal" in table:
        styled = styled.map(lambda v: "" if pd.isna(v) else "color: #d73027" if v < -0.1 else "", subset=["vs normal"])
    st.dataframe(styled, hide_index=True, height=38 + 35 * len(table))
    if notes:
        st.caption("Live traffic unavailable for: " + "; ".join(notes))

    if not balance.empty:
        left, right = st.columns(2)
        show = lambda df: df.assign(**{c: df[c] / per for c in ("Production", "Consumption", "Net")})[
            ["Country", "Production", "Consumption", "Net"]].style.format(
            {c: "{:,.1f}" for c in ("Production", "Consumption", "Net")})
        left.markdown(f"**Largest net exporters ({label})**")
        left.dataframe(show(balance.nlargest(10, "Net")), hide_index=True, height=38 + 35 * 10)
        right.markdown(f"**Largest net importers ({label})**")
        right.dataframe(show(balance.nsmallest(10, "Net")), hide_index=True, height=38 + 35 * 10)
        st.caption(f"Converted from energy units at about {per:g} TWh per {label.split('/')[0]}, so treat as approximate. "
                   "Net balance is production minus consumption: it ignores refined-product trade and stock changes.")
    st.caption("Sources: Energy Institute Statistical Review via Our World in Data (countries); EIA World Oil Transit "
               "Chokepoints, March 2026 (chokepoint volumes); IMF PortWatch (daily tanker transits). PortWatch's "
               "Oresund and Bosporus counts stand in for the Danish and Turkish Straits.")
