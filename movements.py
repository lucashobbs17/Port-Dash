

import pandas as pd
from positions import get_full_positions, get_trades
from risk import get_position_weights
from volatility import get_daily_returns, get_price_history



def get_daily_moves():
    returns = get_daily_returns()
    today = returns.iloc[-1]
    normal_vol = returns.iloc[-61:-1].std()
    value = get_position_weights().set_index('ticker')['value']

    moves = pd.DataFrame({
        'return': today,
        'dollar_move': today * value,
        'z_score': today / normal_vol,
    })
    moves.index.name = 'ticker'
    return moves.reset_index()

def get_drawdowns():
    prices = get_price_history()
    running_high = prices.cummax()
    drawdowns = (prices/running_high - 1)
    return drawdowns.iloc[-1] 

def get_since_purchase():
    trades = get_trades()
    trades['date'] = pd.to_datetime(trades['date'])
    entry_dates = trades[trades['side'] == 'buy'].groupby('ticker')['date'].min()

    avg_price = get_full_positions().set_index('ticker')['avg_price']
    prices = get_price_history(base=None)

    drawdowns = {}
    for ticker, entry in entry_dates.items():
        held = prices.loc[entry:, ticker]
        drawdowns[ticker] = (held / held.cummax() - 1).iloc[-1]

    return pd.DataFrame({
        'entry_date': entry_dates,
        'return_since_purchase': prices.iloc[-1] / avg_price - 1,
        'drawdown_since_purchase': pd.Series(drawdowns),
    })

if __name__ == '__main__':
    daily_moves = get_daily_moves()
    print(daily_moves)  
    drawdowns = get_drawdowns()
    print(drawdowns)
    drawdowns_since = get_since_purchase()
    print(drawdowns_since)
    print(get_trades())
    print(get_full_positions())
