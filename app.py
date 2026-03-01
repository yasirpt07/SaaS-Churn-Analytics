from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load artifacts
model = joblib.load("data/churn_model.pkl")
scaler = joblib.load("data/scaler.pkl")
model_columns = joblib.load("data/model_columns.pkl")

@app.route("/")
def home():
    return "Churn Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Convert incoming JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Apply same dummy encoding
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[model_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)