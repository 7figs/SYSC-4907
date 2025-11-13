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

@app.route("/load_store_delete")
def loadStoreDelete():
    return render_template("load_store.html")

@app.route("/cold_dislike")
def coldDislike():
    return render_template("coldDislike.html")

if __name__ == "__main__":
    app.run(debug=True)
