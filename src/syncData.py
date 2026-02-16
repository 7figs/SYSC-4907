import sqlite3
import const

def syncData(data, id, salt):
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {const.USERS_TABLE} WHERE user_id = ?", (id,))
    test = cursor.fetchall()
    if len(test) == 0:
        cursor.execute(f"INSERT INTO {const.USERS_TABLE} (user_id, data, salt) VALUES(?, ?, ?)", (id, data, salt,))
    else:
        cursor.execute(f"UPDATE {const.USERS_TABLE} SET user_id == ?, salt == ?, data= ? WHERE user_id = ?", (id, salt, data, id))
    conn.commit()
    conn.close()