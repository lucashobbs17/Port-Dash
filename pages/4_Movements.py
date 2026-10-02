import streamlit as st
from movements import get_daily_moves, get_drawdowns, get_since_purchase, get_since_purchase

st.title("Movements & Monitoring")

@st.fragment(run_every="5m")
def live_section():
    moves = get_daily_moves()

    flagged = moves[moves['z_score'].abs() > 2]
    for _, row in flagged.iterrows():
        st.warning(f"{row['ticker']}: unusual move of {row['return']:.2%} (z = {row['z_score']:.1f})")

    st.subheader("Today's moves")
    st.dataframe(
        moves.style.format({'return': '{:.2%}', 'dollar_move': '${:,.2f}', 'z_score': '{:.2f}'}),
        hide_index=True,
    )

    st.subheader("Drawdown from 2-year high")
    st.dataframe(get_drawdowns().to_frame('drawdown').style.format('{:.2%}'))
    st.subheader("Since purchase")
    st.dataframe(get_since_purchase().style.format({
        'entry_date': '{:%Y-%m-%d}',
        'return_since_purchase': '{:.2%}',
        'drawdown_since_purchase': '{:.2%}',
    }))
live_section()