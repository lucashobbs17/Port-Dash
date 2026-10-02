
import sqlite3

conn = sqlite3.connect('trades.db')

cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        ticker TEXT,
        side TEXT,
        quantity REAL,
        price REAL,
        deleted INTEGER DEFAULT 0
    )
''')


cursor.execute('''
    INSERT INTO trades (date, ticker, side, quantity, price)
    VALUES (?, ?, ?, ?, ?)
''', ('2026-09-29', 'AAPL', 'buy', 10, 150.00))

conn.commit()
conn.close()