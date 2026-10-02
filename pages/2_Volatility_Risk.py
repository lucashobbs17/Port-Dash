import streamlit as st
from volatility import get_rolling_volatility, get_vol_risk_summary

st.title("Volatility & Risk")
st.dataframe(get_vol_risk_summary())

rolling_vol20 = get_rolling_volatility(20)
rolling_vol60 = get_rolling_volatility(60)

st.subheader("20-day rolling volatility")
st.line_chart(rolling_vol20.dropna())

st.subheader("60-day rolling volatility")
st.line_chart(rolling_vol60.dropna())