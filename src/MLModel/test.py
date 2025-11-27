import matplotlib.pyplot as plt
from sklearn import tree
import pickle
import sqlite3
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

FEATURES = [
    "release_year",
    "actor_Samuel_L_Jackson",
    "actor_Morgan_Freeman",
    "actor_John_Ratzenberger",
    "actor_Tom_Hanks",
    "actor_Robert_De_Niro",
    "actor_Harrison_Ford",
    "actor_Leonardo_DiCaprio",
    "actor_Brad_Pitt",
    "actor_Christain_Bale",
    "genre_Drama",
    "genre_Comedy",
    "genre_Romance",
    "genre_Thriller",
    "genre_Adventure",
    "genre_Action",
    "genre_Crime",
    "genre_Family",
    "genre_Science_Fiction",
    "genre_Fantasy"
]

CLASSES = [
    "No Recommend",
    "Recommend"
]


# Feature order: [
# release_year, 
# actor_Samuel_L_Jackson, 
# actor_Morgan_Freeman, 
# actor_John_Ratzenberger, 
# actor_Tom_Hanks, 
# actor_Robert_De_Niro, 
# actor_Harrison_Ford, 
# actor_Leonardo_DiCaprio, 
# actor_Brad_Pitt, 
# actor_Christain_Bale, 
# genre_Drama, 
# genre_Comedy, 
# genre_Romance, 
# genre_Thriller, 
# genre_Adventure, 
# genre_Action, 
# genre_Crime, 
# genre_Family, 
# genre_Science_Fiction, 
# genre_Fantasy
# ]
X = []
Y = []
test_samples = []

conn = sqlite3.connect(const.DB_PATH)
cursor = conn.cursor()
cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'The Godfather'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(1)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'The Godfather: Part II'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(1)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'GoodFellas'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(1)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'The Shawshank Redemption'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(1)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'The Good, the Bad and the Ugly'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(1)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Spirited Away'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(0)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Star Wars'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(0)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Inside Out'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(0)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'The Matrix'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(0)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Furious 7'")
test = cursor.fetchall()
X.append(list(test[0][-21:]))
Y.append(0)

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Pulp Fiction'")
test = cursor.fetchall()[0]
test_samples.append(list(test[-21:]))

cursor.execute(f"SELECT * FROM {const.MOVIES_TABLE} WHERE title == 'Big Hero 6'")
test = cursor.fetchall()
test_samples.append(list(test[0][-21:]))
conn.close()

clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, Y)
test = pickle.dumps(clf)
clf = pickle.loads(test)
print(clf.predict(test_samples))

fig = plt.figure(figsize=(10, 8))
tree.plot_tree(clf, impurity=False, filled=True, feature_names=FEATURES, class_names=CLASSES)
plt.savefig('decision_tree.png')
plt.close(fig)