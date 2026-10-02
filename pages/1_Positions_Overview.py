import streamlit as st
from positions import get_positions, get_realized_pnl, get_unrealized_pnl, get_full_positions
# streamlit run app.py

st.title("Positions Overview")
st.dataframe(get_full_positions())

