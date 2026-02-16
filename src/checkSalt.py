import sqlite3
import const
import os

def checkSalt():
    salt = os.urandom(16)
    salt = salt.hex()
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {const.SALTS_TABLE}")
    test = cursor.fetchall()
    while salt in test:
        salt = os.urandom(16)
        salt = salt.hex()
    cursor.execute(f"INSERT INTO {const.SALTS_TABLE} (salt) VALUES(?)", (salt,))
    conn.commit()
    conn.close()
    return salt