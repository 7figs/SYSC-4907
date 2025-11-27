from flask import Flask, render_template, request, jsonify
import sqlite3
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/start")
def coldStart():
    return render_template("coldStart.html")

@app.route("/choose_account")
def chooseAccount():
    return render_template("chooseAccount.html")

@app.route("/settings")
def loadStoreDelete():
    return render_template("settings.html")

@app.route("/video")
def video():
    return render_template("video.html")

"""
endpoints
"""
@app.route("/movies", methods=["GET"])
def getMovies():
    conn = sqlite3.connect(const.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT title,overview FROM {const.MOVIES_TABLE}")
    movies = cursor.fetchall()
    return movies

@app.route("/tree", methods=["GET"])
def createTree():
    like = request.args.get("l")
    dislike = request.args.get("d")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
