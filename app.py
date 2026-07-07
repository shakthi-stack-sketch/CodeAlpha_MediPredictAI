"""
MediPredict AI - Flask Web Application
Serves the healthcare-themed UI and disease risk prediction API.
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

app = Flask(__name__)

# Load model artifacts at startup
model = None
scaler = None
metadata = {}


def load_model_artifacts():
    """Load trained model, scaler, and metadata from disk."""
    global model, scaler, metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please run 'python model_training.py' first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_name": "Random Forest",
            "accuracy": "N/A",
            "dataset": "UCI Heart Disease (Cleveland)",
        }


# Feature definitions for the prediction form
FEATURE_FIELDS = [
    {
        "name": "age",
        "label": "Age",
        "type": "number",
        "min": 18,
        "max": 100,
        "placeholder": "e.g. 55",
        "help": "Patient age in years",
    },
    {
        "name": "sex",
        "label": "Sex",
        "type": "select",
        "options": [
            {"value": "0", "label": "Female"},
            {"value": "1", "label": "Male"},
        ],
        "help": "Biological sex",
    },
    {
        "name": "cp",
        "label": "Chest Pain Type",
        "type": "select",
        "options": [
            {"value": "1", "label": "Typical Angina"},
            {"value": "2", "label": "Atypical Angina"},
            {"value": "3", "label": "Non-anginal Pain"},
            {"value": "4", "label": "Asymptomatic"},
        ],
        "help": "Type of chest pain experienced",
    },
    {
        "name": "trestbps",
        "label": "Resting Blood Pressure",
        "type": "number",
        "min": 80,
        "max": 220,
        "placeholder": "e.g. 130",
        "help": "Resting blood pressure in mm Hg",
    },
    {
        "name": "chol",
        "label": "Cholesterol",
        "type": "number",
        "min": 100,
        "max": 600,
        "placeholder": "e.g. 240",
        "help": "Serum cholesterol in mg/dl",
    },
    {
        "name": "fbs",
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "type": "select",
        "options": [
            {"value": "0", "label": "No"},
            {"value": "1", "label": "Yes"},
        ],
        "help": "Fasting blood sugar above 120 mg/dl",
    },
    {
        "name": "restecg",
        "label": "Resting ECG Results",
        "type": "select",
        "options": [
            {"value": "0", "label": "Normal"},
            {"value": "1", "label": "ST-T Wave Abnormality"},
            {"value": "2", "label": "Left Ventricular Hypertrophy"},
        ],
        "help": "Resting electrocardiographic results",
    },
    {
        "name": "thalach",
        "label": "Maximum Heart Rate",
        "type": "number",
        "min": 60,
        "max": 220,
        "placeholder": "e.g. 150",
        "help": "Maximum heart rate achieved during exercise",
    },
    {
        "name": "exang",
        "label": "Exercise Induced Angina",
        "type": "select",
        "options": [
            {"value": "0", "label": "No"},
            {"value": "1", "label": "Yes"},
        ],
        "help": "Chest pain induced by exercise",
    },
    {
        "name": "oldpeak",
        "label": "ST Depression (Oldpeak)",
        "type": "number",
        "min": 0,
        "max": 10,
        "step": 0.1,
        "placeholder": "e.g. 1.4",
        "help": "ST depression induced by exercise relative to rest",
    },
    {
        "name": "slope",
        "label": "ST Slope",
        "type": "select",
        "options": [
            {"value": "1", "label": "Upsloping"},
            {"value": "2", "label": "Flat"},
            {"value": "3", "label": "Downsloping"},
        ],
        "help": "Slope of the peak exercise ST segment",
    },
    {
        "name": "ca",
        "label": "Major Vessels (Fluoroscopy)",
        "type": "select",
        "options": [
            {"value": "0", "label": "0 vessels"},
            {"value": "1", "label": "1 vessel"},
            {"value": "2", "label": "2 vessels"},
            {"value": "3", "label": "3 vessels"},
        ],
        "help": "Number of major vessels colored by fluoroscopy",
    },
    {
        "name": "thal",
        "label": "Thalassemia",
        "type": "select",
        "options": [
            {"value": "3", "label": "Normal"},
            {"value": "6", "label": "Fixed Defect"},
            {"value": "7", "label": "Reversible Defect"},
        ],
        "help": "Thalassemia blood disorder indicator",
    },
]


@app.route("/")
def home():
    """Home page with project introduction and features."""
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    """Prediction form page."""
    return render_template("predict.html", fields=FEATURE_FIELDS)


@app.route("/about")
def about():
    """About page with dataset, model, and workflow information."""
    return render_template("about.html", metadata=metadata)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept patient data and return disease risk prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided."}), 400

        # Build feature vector in correct order
        feature_names = [f["name"] for f in FEATURE_FIELDS]
        features = []

        for name in feature_names:
            if name not in data or data[name] == "":
                return jsonify({"error": f"Missing required field: {name}"}), 400
            features.append(float(data[name]))

        feature_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)

        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(max(probabilities) * 100)

        risk_label = "Yes" if prediction == 1 else "No"
        risk_level = "High" if prediction == 1 else "Low"

        if prediction == 1:
            recommendation = (
                "The model indicates elevated heart disease risk based on the provided data. "
                "Please consult a healthcare professional for proper evaluation and screening."
            )
        else:
            recommendation = (
                "The model indicates lower heart disease risk based on the provided data. "
                "Maintain a healthy lifestyle and schedule regular check-ups."
            )

        disclaimer = (
            "This prediction is for educational purposes only and should not replace "
            "professional medical advice."
        )

        return jsonify(
            {
                "prediction": risk_label,
                "risk_level": risk_level,
                "confidence": round(confidence, 1),
                "recommendation": recommendation,
                "disclaimer": disclaimer,
            }
        )

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    load_model_artifacts()
    print("MediPredict AI is running at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
