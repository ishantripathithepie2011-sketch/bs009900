from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from datetime import timedelta
import os
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import joblib
except Exception:
    joblib = None

app = Flask(__name__)
app.secret_key = "oblivion-secret-key"
app.permanent_session_lifetime = timedelta(days=7)

# Load model and scaler once when available
model = None
scaler = None

if tf is not None and joblib is not None and os.path.exists("sales_lstm_model.keras") and os.path.exists("sales_scaler.pkl"):
    try:
        model = tf.keras.models.load_model("sales_lstm_model.keras")
        scaler = joblib.load("sales_scaler.pkl")
    except Exception:
        model = None
        scaler = None


def is_logged_in():
    return bool(session.get("user"))


@app.route("/")
@app.route("/sale")
def home():
    return render_template("sale.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form or {}
        email = (data.get("email") or "").strip()

        if not email:
            return jsonify({"success": False, "error": "Email is required."}), 400

        name = (data.get("name") or email.split("@")[0]).strip()
        remembered = bool(data.get("remembered", False))

        session.permanent = remembered
        session["user"] = {
            "name": name.replace(".", " ").replace("_", " ").replace("-", " ").title(),
            "email": email,
            "remembered": remembered,
        }

        next_url = (data.get("next") or request.args.get("next") or "/sale").strip()
        return jsonify({"success": True, "redirect": next_url})

    return send_from_directory(app.root_path, "login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


@app.route("/me")
def me():
    if is_logged_in():
        return jsonify({"loggedIn": True, "user": session["user"]})
    return jsonify({"loggedIn": False})


@app.route("/oblivion.html")
def oblivion_page():
    return send_from_directory(app.root_path, "oblivion.html")


@app.route("/login.html")
def login_page():
    return send_from_directory(app.root_path, "login.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not is_logged_in():
        return jsonify({
            "error": "Please log in to use the sales prediction service.",
            "requiresLogin": True
        }), 401

    try:
        if model is None or scaler is None:
            return jsonify({
                "error": "The sales model is not available in this environment."
            }), 503

        data = request.json

        sales_values = data["sales"]

        if len(sales_values) != 30:
            return jsonify({
                "error": "Exactly 30 sales values are required."
            })

        sales_values = np.array(
            sales_values
        ).reshape(-1, 1)

        scaled = scaler.transform(sales_values)

        X = scaled.reshape(1, 30, 1)

        pred_scaled = model.predict(X, verbose=0)

        prediction = scaler.inverse_transform(
            pred_scaled
        )[0][0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)