"""
MediPredict AI - Model Training Script
Trains a Random Forest classifier on the UCI Heart Disease (Cleveland) dataset.
Run once before starting the Flask app: python model_training.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "heart_disease_raw.data")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "dataset", "heart_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# UCI Cleveland heart disease feature names
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"


def load_and_preprocess_data() -> pd.DataFrame:
    """Load raw UCI data, clean missing values, and create binary target."""
    column_names = FEATURE_COLUMNS + ["num"]

    df = pd.read_csv(
        RAW_DATA_PATH,
        names=column_names,
        na_values="?",
        skipinitialspace=True,
    )

    # Drop rows with missing values (only a few in this dataset)
    df = df.dropna().reset_index(drop=True)

    # Binary classification: 0 = no disease, 1+ = disease present
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Persist cleaned dataset for reference
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_DATA_PATH}")
    print(f"Records: {len(df)} | Disease cases: {df[TARGET_COLUMN].sum()}")

    return df


def train_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and Logistic Regression; return the better model."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
    }

    best_name = None
    best_model = None
    best_accuracy = 0.0
    best_report = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy:.2%}")
        print(report)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
            best_report = report

    return best_name, best_model, best_accuracy, best_report


def save_artifacts(model, scaler, model_name, accuracy, report):
    """Save trained model, scaler, and metadata to the models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "dataset": "UCI Heart Disease (Cleveland)",
        "records": None,
        "classification_report": report,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


def main():
    print("=" * 50)
    print("MediPredict AI - Model Training")
    print("=" * 50)

    df = load_and_preprocess_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified split for balanced classes in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for consistent inference
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_name, model, accuracy, report = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    save_artifacts(model, scaler, model_name, accuracy, report)

    # Update record count in metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["records"] = len(df)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {model_name} ({accuracy:.2%} accuracy)")
    print("Training complete. Run 'python app.py' to start the web app.")


if __name__ == "__main__":
    main()
