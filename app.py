import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and expected feature list
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("feature_names.pkl")  # Must match columns used during training

st.title("🧠 Stroke Risk Prediction App")

# --- User Inputs ---
age = st.slider("Age", 0, 100)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)

gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Initial Raw Input Dictionary ---
raw_input = {
    "age": age,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "gender": gender,
    "ever_married": ever_married,
    "work_type": work_type,
    "residence_type": residence_type,
    "smoking_status": smoking_status,
}

# --- Convert to DataFrame ---
input_df = pd.DataFrame([raw_input])

# --- One-Hot Encode Categorical Columns (drop_first=True like training) ---
categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# --- Align Columns with Training Features ---
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# --- Scale Numerical Columns ---
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# --- Predict ---
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke ({probability:.2%} probability)")
    else:
        st.success(f"✅ Low Risk of Stroke ({probability:.2%} probability)")
