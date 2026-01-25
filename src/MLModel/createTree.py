from sklearn import tree
import pickle
import sqlite3
import sys
import os
import base64

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

def createTree(likes, dislikes):
    X = []
    Y = []
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    for like in likes:
        cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE} WHERE rowid == "{like}"')
        test = cursor.fetchall()
        X.append(list(test[0][-21:]))
        Y.append(1)
    for dislike in dislikes:
        cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE} WHERE rowid == "{dislike}"')
        test = cursor.fetchall()
        X.append(list(test[0][-21:]))
        Y.append(0)
    conn.close()
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, Y)
    clf = pickle.dumps(clf)
    clf = base64.urlsafe_b64encode(clf).decode("ascii")
    return [clf]