import sqlite3

conn = sqlite3.connect("db.db")
cursor = conn.cursor()
cursor.execute("""
        DROP TABLE temp
    """)
conn.commit()
conn.close()