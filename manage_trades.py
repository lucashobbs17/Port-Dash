import sqlite3

DB = "trades.db"

def show_trades():
    with sqlite3.connect(DB) as conn:
        for row in conn.execute("SELECT * FROM trades"):
            print(row)

def soft_delete(trade_id):
    with sqlite3.connect(DB) as conn:
        conn.execute("UPDATE trades SET deleted = 1 WHERE trade_id = ?", (trade_id,))

def add_trade(date, ticker, side, quantity, price):
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO trades (date, ticker, side, quantity, price, deleted) VALUES (?, ?, ?, ?, ?, 0)",
            (date, ticker, side, quantity, price),
        )
def hard_delete(trade_id):
    with sqlite3.connect(DB) as conn:
        conn.execute("DELETE FROM trades WHERE trade_id = ?", (trade_id,))

def rename_ticker(old, new):
    with sqlite3.connect(DB) as conn:
        conn.execute("UPDATE trades SET ticker = ? WHERE ticker = ?", (new, old))
if __name__ == "__main__":

    show_trades()