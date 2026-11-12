import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encoder
model = joblib.load('heart.joblib')


def main():
    st.title('Heart Disease Prediction')
    st.subheader('Patient Information')

    # Input one by one
    age = st.number_input("Age (years)", 1, 100, 50)

    sex = st.radio(
        "Sex",
        ["Male", "Female"]
    )

    dataset = st.selectbox(
        "Dataset",
        ["Cleveland", "Hungary", "Switzerland", "Long Beach VA"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)", 50, 250, 120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)", 50, 700, 200
    )

    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    restecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )

    thalch = st.number_input(
        "Maximum Heart Rate Achieved", 50, 250, 150
    )

    exang = st.radio(
        "Exercise-Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1
    )

    # Create input dictionary
    data = {
        'age': int(age),
        'sex': sex,
        'dataset': dataset,
        'cp': cp,
        'trestbps': int(trestbps),
        'chol': int(chol),
        'fbs': fbs,
        'restecg': restecg,
        'thalch': int(thalch),
        'exang': exang,
        'oldpeak': float(oldpeak)
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    if st.button("Make Prediction", type="primary"):

        prediction = model.predict(df)[0]

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("Prediction: No Heart Disease")
        else:
            st.error("Prediction: Heart Disease Detected")


if __name__ == "__main__":
    main()