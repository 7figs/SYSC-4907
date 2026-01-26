from sklearn import tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import sqlite3
import sys
import os
import base64

FEATURES = [
    "release_year",
    "actor_Samuel_L_Jackson",
    "actor_Morgan_Freeman",
    "actor_John_Ratzenberger",
    "actor_Tom_Hanks",
    "actor_Robert_De_Niro",
    "actor_Harrison_Ford",
    "actor_Leonardo_DiCaprio",
    "actor_Matt_Damon",
    "actor_Brad_Pitt",
    "actor_Christain_Bale",
    "genre_Action",
    "genre_Adventure",
    "genre_Animation",
    "genre_Comedy",
    "genre_Crime",
    "genre_Documentary",
    "genre_Drama",
    "genre_Family",
    "genre_Fantasy",
    "genre_Foreign",
    "genre_History",
    "genre_genre_Horror",
    "genre_Music",
    "genre_Mystery",
    "genre_Romance",
    "genre_Science_Fiction",
    "genre_TV_Movie",
    "genre_Thriller",
    "genre_War",
    "genre_Western"
]

CLASSES = [
    "No Recommend",
    "Recommend"
]

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
        X.append(list(test[0][2:]))
        print(list(test[0][2:]))
        Y.append(1)
    for dislike in dislikes:
        cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE} WHERE rowid == "{dislike}"')
        test = cursor.fetchall()
        X.append(list(test[0][2:]))
        Y.append(0)
    conn.close()
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, Y)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    tree.plot_tree(
        clf,
        ax=ax,
        impurity=False,
        filled=True,
        feature_names=FEATURES,
        class_names=CLASSES
    )

    fig.savefig("test.png", bbox_inches="tight")
    plt.close(fig)

    clf = pickle.dumps(clf)
    clf = base64.urlsafe_b64encode(clf).decode("ascii")
    return [clf]