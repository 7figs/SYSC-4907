from flask import Flask, render_template, request
import sqlite3
import json
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
target_dir = os.path.abspath(os.path.join("../MLModel"))
sys.path.insert(0, parent_dir)
import const
sys.path.insert(0, target_dir)
import createTree as ml_create_tree
import recommend as ml_recommend
import createUserVector as ml_user_vector

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/start")
def coldStart():
    return render_template("coldStart.html")

@app.route("/settings")
def loadStoreDelete():
    return render_template("settings.html")

@app.route("/watch/<id>/<name>")
def watch(id, name):
    return render_template("watch.html")

@app.route("/feed/<id>")
def feed(id):
    return render_template("feed.html")

@app.route("/settings/<id>")
def settings(id):
    return render_template("settings.html")

@app.route("/preview/<id>/<name>")
def preview(id, name):
    return render_template("preview.html")

"""
Endpoints
"""
@app.route("/movies", methods=["GET"])
def getMovies():
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT rowid, title, overview FROM {const.MOVIES_TABLE}")
    movies = cursor.fetchall()
    conn.close()
    return movies

@app.route("/tree", methods=["GET"])
def createTree():
    like = (request.args.get("l"))
    dislike = (request.args.get("d"))
    like = json.loads(like)
    dislike = json.loads(dislike)
    tree = ml_create_tree.createTree(like, dislike)
    return tree

@app.route("/recommend", methods=["GET"])
def recommend():
    tree = (request.args.get("t"))
    user_vector = (request.args.get("v"))
    too_soon = (request.args.get("s"))
    tree = json.loads(tree)
    user_vector = json.loads(user_vector)
    too_soon = json.loads(too_soon)
    recommendations = ml_recommend.recommend(tree, user_vector, too_soon)
    return recommendations

@app.route("/user-vector", methods=["GET"])
def user_vector():
    history = (request.args.get("h"))
    preferences = (request.args.get("p"))
    days = (request.args.get("d"))
    history = json.loads(history)
    preferences = json.loads(preferences)
    days = json.loads(days)
    user_vector = ml_user_vector.createUserVector(history, preferences, days)
    return user_vector

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")