from sklearn import tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import sqlite3
import sys
import os
import base64
from io import BytesIO
from matplotlib.patches import Rectangle, FancyArrowPatch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

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
        Y.append(1)
    for dislike in dislikes:
        cursor.execute(f'SELECT * FROM {const.MOVIES_TABLE} WHERE rowid == "{dislike}"')
        test = cursor.fetchall()
        X.append(list(test[0][2:]))
        Y.append(0)
    conn.close()
    clf = tree.DecisionTreeClassifier(
        random_state= 42,
        max_depth= None,
        min_samples_leaf= 0.2,
        class_weight= {0.0: 1, 1.0: 3},
    )
    clf = clf.fit(X, Y)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    tree.export_graphviz(clf)
    tree.plot_tree(
        clf,
        ax=ax,
        impurity=False,
        filled=True,
        feature_names=const.FEATURES,
        class_names=const.CLASSES
    )
    for patch in ax.patches:
        if isinstance(patch, plt.FancyBboxPatch):
            patch.set_edgecolor("white")
            patch.set_linewidth(2)
    
    for line in ax.get_lines():
        line.set_color("white")
        line.set_linewidth(2)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    clf = pickle.dumps(clf)
    clf = base64.urlsafe_b64encode(clf).decode("ascii")
    return [clf, img_base64]