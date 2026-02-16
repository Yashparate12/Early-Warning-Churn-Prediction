from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    features = np.array([
        float(data["tenure"]),
        float(data["monthly_charges"]),
        float(data["total_charges"])
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    risk = "HIGH RISK" if probability > 0.6 else "LOW RISK"

    return render_template(
        "dashboard.html",
        churn=int(prediction),
        probability=round(probability * 100, 2),
        risk=risk
    )

if __name__ == "__main__":
    app.run(debug=True)
