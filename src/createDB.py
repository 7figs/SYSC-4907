import sqlite3

conn = sqlite3.connect("db.db")
cursor = conn.cursor()
cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            overview TEXT NOT NULL,
            release_year INTEGER NOT NULL,
            actor_Samuel_L_Jackson INTEGER NOT NULL,
            actor_Morgan_Freeman INTEGER NOT NULL,
            actor_John_Ratzenberger INTEGER NOT NULL,
            actor_Tom_Hanks INTEGER NOT NULL,
            actor_Robert_De_Niro INTEGER NOT NULL,
            actor_Harrison_Ford INTEGER NOT NULL,
            actor_Leonardo_DiCaprio INTEGER NOT NULL,
            actor_Brad_Pitt INTEGER NOT NULL,
            actor_Christain_Bale INTEGER NOT NULL,
            genre_Drama INTEGER NOT NULL,
            genre_Comedy INTEGER NOT NULL,
            genre_Romance INTEGER NOT NULL,
            genre_Thriller INTEGER NOT NULL,
            genre_Adventure INTEGER NOT NULL,
            genre_Action INTEGER NOT NULL,
            genre_Crime INTEGER NOT NULL,
            genre_Family INTEGER NOT NULL,
            genre_Science_Fiction INTEGER NOT NULL,
            genre_Fantasy INTEGER NOT NULL
        )
    """)
conn.commit()
conn.close()