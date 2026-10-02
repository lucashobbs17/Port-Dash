import sqlite3
import pandas as pd
import yfinance as yf
import streamlit as st


def get_trades():
    conn = sqlite3.connect('trades.db')
    trades_df = pd.read_sql('SELECT * FROM trades WHERE deleted = 0', conn)
    conn.close()
    return trades_df


def signed_quantity(row):
    if row['side'] == 'buy':
        return row['quantity']
    elif row['side'] == 'sell':
        return -row['quantity']


def get_positions():
    trades_df = get_trades()
    trades_df['signed_quantity'] = trades_df.apply(signed_quantity, axis=1)

    quantity_df = trades_df.groupby('ticker').agg({
        'signed_quantity': 'sum'
    }).reset_index()

    buys_df = trades_df[trades_df['side'] == 'buy']
    avg_price = buys_df.groupby('ticker').apply(
        lambda g: (g['price'] * g['quantity']).sum() / g['quantity'].sum()
    ).reset_index(name='avg_price')

    positions_df = quantity_df.merge(avg_price, on='ticker')
    return positions_df


def get_realized_pnl():
    trades_df = get_trades()
    sorted_trades_df = trades_df.sort_values(by='date')
    grouped = sorted_trades_df.groupby('ticker')
    results = []
    for ticker, group in grouped:
        running_quantity = 0
        running_avprice = 0.0
        for index, row in group.iterrows():
            if row['side']== 'buy':
                running_quantity += row['quantity']
                running_avprice = (running_avprice * (running_quantity - row['quantity']) + row['price'] * row['quantity']) / running_quantity
            elif row['side'] == 'sell':
                realized_pnl = (row['price'] - running_avprice) * row['quantity']
                running_quantity -= row['quantity']
                results.append({'ticker': ticker, 'date': row['date'], 'realized_pnl': realized_pnl})
                if running_quantity == 0:
                    running_avprice = 0.0
                else:
                    running_avprice = (running_avprice * (running_quantity + row['quantity']) - row['price'] * row['quantity']) / running_quantity

    return pd.DataFrame(results)

def get_unrealized_pnl():
    from volatility import get_price_history
    positions_df = get_positions()
    latest = get_price_history(base=None).iloc[-1]
    current_price = positions_df['ticker'].map(latest)
    positions_df['unrealized_pnl'] = (current_price - positions_df['avg_price']) * positions_df['signed_quantity']
    return positions_df


def get_full_positions():
    unrealized_df = get_unrealized_pnl()
    realized_df = get_realized_pnl()
    realized_summary = realized_df.groupby('ticker')['realized_pnl'].sum().reset_index()
    full_df = unrealized_df.merge(realized_summary, on='ticker', how='left')
    full_df['realized_pnl'] = full_df['realized_pnl'].fillna(0)
    return full_df

if __name__ == '__main__':
    print(get_positions())
    print(get_trades())
    print(get_realized_pnl())
    print(get_full_positions())