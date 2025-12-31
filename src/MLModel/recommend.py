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

def recommend(tree):
    clf = tree.encode("ascii")
    clf = base64.urlsafe_b64decode(clf)
    clf = pickle.loads(clf)

    recommendations = []

    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE}')
    movies = cursor.fetchall()
    for movie in movies:
        result = clf.predict([list(movie[-21:])])
        if result:
            recommendations.append(movie[0])
    conn.close()

    return random.sample(recommendations, k=10)
