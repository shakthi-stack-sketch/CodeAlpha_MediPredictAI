# MediPredict AI

**AI-Powered Disease Risk Prediction**

MediPredict AI is a lightweight machine learning web application that predicts heart disease risk using patient medical data. Built with Python, Flask, and Scikit-learn.

## Features

- **Home Page** — Project introduction, features overview, and quick access to prediction
- **Prediction Page** — Enter clinical measurements and receive instant risk assessment
- **About Page** — Dataset, model, and workflow documentation
- **Random Forest Classifier** — Trained on the UCI Heart Disease (Cleveland) dataset
- **Confidence Scoring** — Probability-based confidence for each prediction
- **Healthcare UI** — Clean, modern, responsive design

## Project Structure

```
CodeAlpha_MediPredictAI/
│
├── app.py                  # Flask web application
├── model_training.py       # Model training script
├── requirements.txt        # Python dependencies
├── README.md
├── dataset/
│   ├── heart_disease_raw.data   # UCI Cleveland raw dataset
│   └── heart_disease.csv        # Cleaned dataset (generated)
├── models/
│   ├── heart_disease_model.joblib   # Trained model (generated)
│   ├── scaler.joblib                # Feature scaler (generated)
│   └── model_metadata.json          # Model info (generated)
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── predict.html
│   └── about.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python model_training.py
```

This downloads and processes the UCI Heart Disease dataset, trains Random Forest and Logistic Regression models, selects the best performer, and saves artifacts to the `models/` folder.

### 3. Start the Web Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

## Input Features

The prediction form accepts these clinical measurements:

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | Male or Female |
| Chest Pain Type | Type of chest pain experienced |
| Resting Blood Pressure | In mm Hg |
| Cholesterol | Serum cholesterol in mg/dl |
| Fasting Blood Sugar | Above 120 mg/dl (Yes/No) |
| Resting ECG | Electrocardiographic results |
| Maximum Heart Rate | Achieved during exercise |
| Exercise Induced Angina | Yes/No |
| ST Depression (Oldpeak) | Induced by exercise |
| ST Slope | Peak exercise ST segment slope |
| Major Vessels | Colored by fluoroscopy (0-3) |
| Thalassemia | Blood disorder indicator |

## Dataset

**UCI Heart Disease (Cleveland)** — A public dataset from the UCI Machine Learning Repository containing patient records with 13 clinical features and heart disease diagnosis labels.

Source: https://archive.ics.uci.edu/ml/datasets/heart+disease

## Machine Learning

- **Algorithm:** Random Forest Classifier (auto-selected if best accuracy)
- **Preprocessing:** StandardScaler feature normalization
- **Validation:** 80/20 stratified train-test split
- **Target:** Binary classification (Disease Risk: Yes/No)

## Disclaimer

This application is for **educational purposes only**. Predictions should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Technologies

Python · Flask · Scikit-learn · Pandas · NumPy · HTML · CSS · JavaScript
