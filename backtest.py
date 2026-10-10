
import pandas as pd
import numpy as np


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame,
                 cost_bps: float = 5) -> pd.Series:
    returns = (prices.pct_change().fillna(0) * weights.shift(1)).sum(axis=1)
    trades = weights.diff().fillna(weights)
    return returns - cost_bps / 10000 * (trades.abs().sum(axis=1))

def get_metrics(returns: pd.Series, rf: float = 0.04) -> dict:
    volatility = returns.std() * np.sqrt(252)
    ann_return = returns.mean() * 252
    sharpe = (ann_return - rf) / volatility if volatility != 0 else np.nan
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max - 1)
    return {'ann_return': ann_return, 'ann_vol': volatility, 'sharpe': sharpe, 'max_drawdown': drawdowns.min()}

def print_metrics(name: str, m: dict) -> None:
    print(f"{name:<20} return {m['ann_return']:>7.1%}   vol {m['ann_vol']:>6.1%}   "
          f"sharpe {m['sharpe']:>5.2f}   max DD {m['max_drawdown']:>7.1%}")

    
if __name__ == "__main__":
    from streamlit import logger
    logger.set_log_level("ERROR")

    from strategy1 import momentum
    from volatility import get_price_history
    from universes import SECTORS 

    prices = get_price_history(period='10y', tickers=SECTORS)

    for name, w in [("Ungrouped", momentum(prices)),
                    ("Grouped", momentum(prices, groups=SECTORS))]:
        for bps in (0, 5):
            r = run_backtest(w, prices, cost_bps=bps)
            print_metrics(f"{name} {bps}bps", get_metrics(r, rf=0))
        turnover = w.diff().abs().sum(axis=1).mean()
        print(f"{name} turnover: {turnover:.1%}")

