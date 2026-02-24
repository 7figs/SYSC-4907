import sqlite3

conn = sqlite3.connect("db.db")
cursor = conn.cursor()
cursor.execute("""
        DELETE FROM salts
    """)
conn.commit()
conn.close()