import sqlite3

conn = sqlite3.connect('trades.db')
cursor = conn.cursor()

cursor.execute('''
    INSERT INTO trades (date, ticker, side, quantity, price)
    VALUES (?, ?, ?, ?, ?)
''', ('2026-09-29', 'MSFT', 'buy', 5, 500.00))

conn.commit()
conn.close()