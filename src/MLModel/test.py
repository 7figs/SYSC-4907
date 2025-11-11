import matplotlib.pyplot as plt
from sklearn import tree
import pickle

FEATURES = [
    "isAction",
    "isComedy",
    "isDrama",
    "isHorror",
    "isScifi",
    "isRomance",
    "isFantasy",
    "isAnimation",
    "isAdventure"
]

CLASSES = [
    "No Recommend",
    "Recommend"
]

# Feature order: [isAction, isComedy, isDrama, isHorror, isScifi, isRomance, isFantasy, isAnimation, isAdventure]
X = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # Citizen Kane
    [0, 0, 1, 0, 0, 1, 0, 0, 0], # Cassablanca
    [1, 0, 1, 0, 0, 0, 0, 0, 0], # The Godfather
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # Raging Bull
    [0, 1, 0, 0, 0, 1, 0, 0, 0], # Singin' in the Rain
    [0, 0, 1, 0, 0, 1, 0, 0, 0], # Gone with the wind
    [0, 0, 1, 0, 0, 0, 0, 0, 1], # Larence of Arabia
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # Schindler's List
    [0, 0, 1, 1, 0, 1, 0, 0, 0], # Vertigo
    [0, 0, 0, 0, 0, 0, 1, 0, 1], # Wizard of Oz
    [0, 1, 1, 0, 0, 1, 0, 0, 0], # City Lights
    [1, 0, 1, 0, 0, 0, 0, 0, 1], # The Searchers
    [1, 0, 0, 0, 1, 0, 1, 0, 1], # Star Wars: A New Hope
    [0, 0, 0, 1, 0, 0, 0, 0, 0], # Psycho
    [0, 0, 1, 0, 1, 0, 0, 0, 1], # 2001: A Space Odyssey
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # Sunset Boulevard
    [0, 1, 1, 0, 0, 1, 0, 0, 0], # The Graduate
    [1, 1, 0, 0, 0, 0, 0, 0, 1], # The General
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # On the Waterfront
    [0, 0, 1, 0, 0, 0, 1, 0, 0], # It's a Wonderful Life
    [0, 0, 1, 1, 0, 0, 0, 0, 0], # Cinatown
    [0, 1, 0, 0, 0, 1, 0, 0, 0], # Some Like It Hot
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # The Grapes of Wrath
    [0, 0, 0, 0, 1, 0, 0, 0, 1], # E.T. the Extra-Terrestrial
    [0, 0, 1, 0, 0, 0, 0, 0, 0]  # To Kill a Mockingbird
]
Y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, Y)
test = pickle.dumps(clf)
clf = pickle.loads(test)
print(clf.predict([
    [1, 0, 1, 1, 0, 0, 0, 0, 0], # Pulp Fiction
    [0, 1, 0, 0, 0, 0, 0, 1, 1] # Toy Story
]))

fig = plt.figure(figsize=(10, 8))
tree.plot_tree(clf, impurity=False, filled=True, feature_names=FEATURES, class_names=CLASSES)
plt.savefig('decision_tree.png')
plt.close(fig)