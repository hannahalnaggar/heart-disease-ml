import streamlit as st
import numpy as np
import joblib

# Load your saved model
model = joblib.load("../models/final_model_tuned.pkl")

# Streamlit page setup
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient data below to predict heart disease risk.")

# --- User inputs ---
sex = st.selectbox("Sex", ["Male", "Female"])
thal = st.selectbox("Thalassemia", [0, 2])  # based on columns thal_0, thal_2
ca = st.selectbox("Number of major vessels (ca)", [0, 2])
slope = st.selectbox("Slope", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [2, 3])
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

# --- One-hot encode based on training columns ---
features = {
    'sex_0': 0, 'sex_1': 0,
    'thal_0': 0, 'thal_2': 0,
    'ca_0': 0, 'ca_2': 0,
    'slope_0': 0, 'slope_1': 0,
    'cp_2': 0, 'cp_3': 0,
    'exang_1': 0
}

# Assign based on user choices
features[f'sex_{1 if sex=="Male" else 0}'] = 1
features[f'thal_{thal}'] = 1
features[f'ca_{ca}'] = 1
features[f'slope_{slope}'] = 1
features[f'cp_{cp}'] = 1
if exang == 1:
    features['exang_1'] = 1

# Arrange in correct order
ordered_features = ['sex_1', 'thal_0', 'ca_2', 'thal_2', 'ca_0',
                    'slope_1', 'cp_2', 'slope_0', 'exang_1', 'sex_0', 'cp_3']

input_data = np.array([[features[col] for col in ordered_features]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})" if prob else "⚠️ High Risk of Heart Disease")
    else:
        st.success(f"✅ No Heart Disease Detected (Probability: {prob:.2f})" if prob else "✅ No Heart Disease Detected")
