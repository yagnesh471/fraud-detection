from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("../models/fraud_detection_model.pkl")
scaler = joblib.load("../models/fraud_scaler.pkl")

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fraud Detection ML API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Convert input into DataFrame
    transaction_df = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    # Scale transaction
    transaction_scaled = scaler.transform(transaction_df)

    # Get fraud probability
    fraud_probability = model.predict_proba(
        transaction_scaled
    )[0][1]

    # Get prediction
    prediction = model.predict(
        transaction_scaled
    )[0]

    result = "Fraud" if prediction == 1 else "Legitimate"

    return jsonify({
        "prediction": result,
        "fraudProbability": float(fraud_probability)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)