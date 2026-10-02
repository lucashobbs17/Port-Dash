import streamlit as st
from risk import get_portfolio_beta, get_position_weights, get_correlation_matrix, get_outright_var, get_portfolio_var

st.title("Portfolio Risk")

st.subheader("Position weights")
st.dataframe(get_position_weights().style.format({'weight': '{:.1%}', 'value': '${:,.2f}'}), hide_index=True)

st.subheader("Correlation matrix")
st.dataframe(get_correlation_matrix(), hide_index=True)

undiversified = get_outright_var().sum()
portfolio = get_portfolio_var()

st.subheader("1-day VaR (95%)")
col1, col2, col3 = st.columns(3)
col1.metric("Undiversified VaR", f"${undiversified:,.2f}")
col2.metric("Portfolio VaR", f"${portfolio:,.2f}")
col3.metric("Diversification benefit", f"${undiversified - portfolio:,.2f}")

st.subheader("Market exposure")
st.metric("Portfolio beta", f"{get_portfolio_beta():.2f}")