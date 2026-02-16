import sqlite3

conn = sqlite3.connect("db.db")
cursor = conn.cursor()
cursor.execute("""
        DELETE FROM users
    """)
conn.commit()
conn.close()