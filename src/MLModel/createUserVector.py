from sklearn import tree
import pickle
import sqlite3
import sys
import os
import base64
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

def createUserVector(history, preferences, days):
    vector = []
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    for i in range(len(history)):
        cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE} WHERE rowid == "{history[i]}"')
        test = cursor.fetchall()
        if len(vector) == 0:
            movie = list(test[0][22:42])
            for genre in movie:
                genre *= (0.5**(days[i] / 7)) * (preferences[i])
            vector = movie
        else:
            movie = list(test[0][22:42])
            for genre in movie:
                genre *= (0.5**(days[i] / 7))
            for j in range(len(movie)):
                vector[j] += movie[j] * (preferences[i])
    return vector
