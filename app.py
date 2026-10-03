"""Streamlit web app for the NFL prop edge finder.

Visitors see the latest saved results. Pulling fresh data spends Odds API credits,
so the Refresh button is protected by a password you set in Streamlit's Secrets.
"""
import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import prop_edge_finder as pef

st.set_page_config(page_title="NFL Prop Edge Finder", layout="wide")


def secret(name):
    """Read a value from Streamlit Secrets (or an environment variable when testing locally)."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


pef.ODDS_KEY = secret("ODDS_API_KEY")
APP_PASSWORD = secret("APP_PASSWORD")

MARKET_NAMES = {
    "player_pass_yds": "Passing yards",
    "player_pass_tds": "Passing TDs",
    "player_rush_yds": "Rushing yards",
    "player_reception_yds": "Receiving yards",
    "player_receptions": "Receptions",
}


@st.cache_resource
def store():
    """One shared box that keeps the latest results for everyone until the app restarts."""
    return {"df": None, "when": None, "skipped": 0, "missed": 0}


def build_df(results):
    rows = [
        {
            "Player": pl, "Market": MARKET_NAMES.get(mk, mk), "Pick": f"{side} {line}",
            "Odds": price, "Line": line, "Projection": round(mu, 2),
            "Edge %": round(edge * 100, 1), "EV per $1": round(ev, 2),
            "Games vs opp": n, "Opponent": opp,
        }
        for edge, ev, pl, mk, side, line, price, mu, n, opp in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Gap %"] = ((df["Projection"] - df["Line"]).abs() / df["Line"].clip(lower=1) * 100).round(0)
    return df


s = store()

st.title("NFL Prop Edge Finder")
st.caption("FanDuel player props vs. recent form and head-to-head history. "
           "A research tool, not betting advice. The model is untested, and big edges "
           "usually mean injury or role news the model can't see.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Filters")
    min_edge = st.slider("Minimum edge (%)", 3, 30, 5)
    max_gap = st.slider("Max gap between line and projection (%)", 10, 100, 35,
                        help="Big gaps usually mean the book knows about an injury or a backup starting.")
    markets = st.multiselect("Markets", list(MARKET_NAMES.values()), default=list(MARKET_NAMES.values()))
    search = st.text_input("Search player")

    st.header("Refresh data")
    if not APP_PASSWORD:
        st.warning("No APP_PASSWORD set in Secrets, so anyone can spend your Odds API credits.")
    entered = st.text_input("Password", type="password")
    can_refresh = (not APP_PASSWORD) or entered == APP_PASSWORD
    if st.button("Refresh now (uses Odds API credits)", disabled=not can_refresh):
        if not pef.ODDS_KEY:
            st.error("ODDS_API_KEY is missing from Secrets.")
        else:
            with st.spinner("Pulling lines and player history. This takes a few minutes..."):
                try:
                    results, skipped, missed = pef.find_edges(min_edge=0.02, max_gap=99, log=lambda m: None)
                    s["df"], s["skipped"], s["missed"] = build_df(results), skipped, len(missed)
                    s["when"] = dt.datetime.now(ZoneInfo("America/New_York"))
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

# ---------- main page ----------
df = s["df"]
if df is None or df.empty:
    st.info("No data yet. Enter the password in the sidebar and press Refresh.")
    st.stop()

st.write(f"Last updated: **{s['when']:%a %b %d, %I:%M %p} ET**. "
         f"{s['missed']} players couldn't be matched to a roster.")

view = df[(df["Edge %"] >= min_edge) & (df["Gap %"] <= max_gap) & (df["Market"].isin(markets))]
if search:
    view = view[view["Player"].str.contains(search, case=False)]
view = view.sort_values("Edge %", ascending=False)
view = view.assign(**{"Check news": view["Edge %"].apply(lambda e: "Yes" if e > 20 else "")})

st.write(f"**{len(view)} props** match your filters.")
st.dataframe(view.drop(columns=["Gap %"]), use_container_width=True, hide_index=True,
             column_config={"Edge %": st.column_config.NumberColumn(format="%.1f%%"),
                            "Odds": st.column_config.NumberColumn(format="%+d")})
