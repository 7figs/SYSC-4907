import sqlite3
import const

def loadData(id):
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {const.USERS_TABLE} WHERE user_id == ?", (id,))
    test = cursor.fetchall()
    if len(test) == 0:
        data = ""
    else:
        data = [test[0][2], test[0][3]]
    conn.commit()
    conn.close()
    return data