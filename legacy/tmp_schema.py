import sqlite3
conn=sqlite3.connect('data/stock.db')
cur=conn.cursor()
cur.execute(\"select name from sqlite_master where type='table' order by name\")
tables=[r[0] for r in cur.fetchall()]
print('tables',tables)
for t in ['stock_daily','stock_5m','stock_5min','stock_five_min','stock_min5']:\n    pass
