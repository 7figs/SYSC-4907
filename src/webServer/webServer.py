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
import createTree as ml

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

@app.route("/video")
def video():
    return render_template("video.html")

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
    cursor.execute(f"SELECT title, overview FROM {const.MOVIES_TABLE}")
    movies = cursor.fetchall()
    conn.close()
    return movies

@app.route("/tree", methods=["GET"])
def createTree():
    like = (request.args.get("l"))
    dislike = (request.args.get("d"))
    like = json.loads(like)
    dislike = json.loads(dislike)
    tree = ml.createTree(like, dislike)
    print(tree)
    return tree

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)