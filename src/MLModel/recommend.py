from sklearn import tree
import pickle
import sqlite3
import sys
import os
import base64
import random
from operator import itemgetter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

def recommend(tree, user_vector, too_soon):
    clf = tree.encode("ascii")
    clf = base64.urlsafe_b64decode(clf)
    clf = pickle.loads(clf)

    recommendations = []
    top_recommendations = []

    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE}')
    movies = cursor.fetchall()
    for movie in movies:
        result = clf.predict([list(movie[2:])])
        if result:
            recommendations.append([movies.index(movie)+1, movie[0]])
            top_recommendations.append([movies.index(movie)+1, movie[0], movie[13:]])
    conn.close()

    if len(user_vector) > 0:
        for i in range(len(top_recommendations)):
            score = 0
            for j in range(len(top_recommendations[i][2])):
                score += user_vector[j] * top_recommendations[i][2][j]
            top_recommendations[i][2] = score

        top_recommendations = sorted(top_recommendations, key=itemgetter(2)) 
        top_recommendations.reverse()
        shift = []
        for item in too_soon:
            for i in range(len(top_recommendations)):
                if top_recommendations[i][0] in too_soon:
                    temp = top_recommendations.pop(i)
                    shift.append(temp)
                    break
        for item in shift:
            top_recommendations.append(item)
        return top_recommendations[:10]
    else:
        for item in too_soon:
            for i in range(len(top_recommendations)):
                if top_recommendations[i][0] in too_soon:
                    temp = top_recommendations.pop(i)
                    break
        return random.sample(recommendations, k=10)
