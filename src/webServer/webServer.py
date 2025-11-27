from flask import Flask, render_template

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
