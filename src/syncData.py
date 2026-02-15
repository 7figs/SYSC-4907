import sqlite3
import const

def syncData(data, id):
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {const.USERS_TABLE} WHERE id == {id}")
    test = cursor.fetchall()
    try:
        if len(test) == 0:
            cursor.execute(f"INSERT INTO {const.USERS_TABLE} (id, data) VALUES({id}, {data})")
        else:
            cursor.execute(f"UPDATE {const.USERS_TABLE} SET id = {id}, data= {data} WHERE id == {id}")
        conn.commit()
        conn.close()
        return True
    except:
        return False