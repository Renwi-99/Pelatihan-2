
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load ML pipeline
@st.cache_resource
def load_model():
    return joblib.load("heart.joblib")

model = load_model()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("❤️ HeartCare AI")

    st.caption("Machine Learning Prediction System")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Patient Assessment", "About the Model"]
    )

    st.divider()

    st.info(
        "This application uses a machine learning "
        "pipeline to estimate heart disease risk."
    )

    st.caption("Version 1.0.0")


# ==========================================
# PATIENT ASSESSMENT
# ==========================================

if page == "Patient Assessment":

    st.markdown(
        '<p class="main-title">Heart Disease Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enter patient information to generate a prediction.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    # ======================================
    # INPUT FORM
    # ======================================

    with st.form("heart_disease_form"):

        # ----------------------------------
        # 1. DEMOGRAPHIC INFORMATION
        # ----------------------------------

        st.markdown(
            '<p class="section-title">👤 Patient Information</p>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=50
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

        with col3:
            dataset = st.selectbox(
                "Dataset Source",
                [
                    "Cleveland",
                    "Hungary",
                    "Switzerland",
                    "Long Beach VA"
                ]
            )

        st.divider()

        # ----------------------------------
        # 2. CLINICAL INFORMATION
        # ----------------------------------

        st.markdown(
            '<p class="section-title">🩺 Clinical Measurements</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            cp = st.selectbox(
                "Chest Pain Type",
                [
                    "typical angina",
                    "atypical angina",
                    "non-anginal",
                    "asymptomatic"
                ]
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50,
                max_value=250,
                value=120
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=50,
                max_value=700,
                value=200
            )

        with col2:

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                ["False", "True"]
            )

            restecg = st.selectbox(
                "Resting ECG Results",
                [
                    "normal",
                    "st-t abnormality",
                    "lv hypertrophy"
                ]
            )

        st.divider()

        # ----------------------------------
        # 3. EXERCISE TEST INFORMATION
        # ----------------------------------

        st.markdown(
            '<p class="section-title">🏃 Exercise Test Results</p>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            thalch = st.number_input(
                "Maximum Heart Rate",
                min_value=50,
                max_value=250,
                value=150
            )

        with col2:
            exang = st.selectbox(
                "Exercise-Induced Angina",
                ["False", "True"]
            )

        with col3:
            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        st.divider()

        # ----------------------------------
        # SUBMIT
        # ----------------------------------

        submitted = st.form_submit_button(
            "🔍 Generate Prediction",
            type="primary",
            use_container_width=True
        )

    # ======================================
    # PREDICTION
    # ======================================

    if submitted:

        # Create input DataFrame
        data = {
            "age": int(age),
            "sex": sex,
            "dataset": dataset,
            "cp": cp,
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": fbs,
            "restecg": restecg,
            "thalch": int(thalch),
            "exang": exang,
            "oldpeak": float(oldpeak)
        }

        df = pd.DataFrame([data])

        try:

            prediction = model.predict(df)[0]

            st.session_state["prediction"] = prediction
            st.session_state["patient_data"] = data

            # Get prediction probability if available
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(df)[0]

                classes = list(model.classes_)

                positive_index = (
                    classes.index(1) if 1 in classes else None
                )

                if positive_index is not None:
                    st.session_state["probability"] = (
                        probabilities[positive_index] * 100
                    )

            else:
                st.session_state.pop("probability", None)

        except Exception as e:

            st.error(f"Prediction error: {e}")

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    if "prediction" in st.session_state:

        st.divider()

        st.markdown("## 📊 Prediction Results")

        prediction = st.session_state["prediction"]

        if prediction == 0:

            st.success(
                "### Low Risk Classification\n\n"
                "The model predicts that heart disease "
                "was not detected."
            )

        else:

            st.error(
                "### Positive Classification\n\n"
                "The model predicts a potential presence "
                "of heart disease."
            )

        # Probability
        if "probability" in st.session_state:

            probability = st.session_state["probability"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Positive Probability",
                    f"{probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Predicted Negative Probability",
                    f"{100 - probability:.2f}%"
                )

            st.progress(
                min(max(probability / 100, 0.0), 1.0)
            )

        # Patient summary
        with st.expander("View Patient Information"):

            patient_df = pd.DataFrame(
                st.session_state["patient_data"].items(),
                columns=["Feature", "Value"]
            )

            st.dataframe(
                patient_df,
                use_container_width=True,
                hide_index=True
            )

        if st.button("🔄 New Assessment"):

            st.session_state.pop("prediction", None)
            st.session_state.pop("patient_data", None)
            st.session_state.pop("probability", None)

            st.rerun()

        st.caption(
            "Disclaimer: This prediction is for educational "
            "and decision-support purposes only. It is not "
            "a medical diagnosis. Consult a qualified "
            "healthcare professional for clinical evaluation."
        )


# ==========================================
# ABOUT THE MODEL
# ==========================================

elif page == "About the Model":

    st.title("About the Model")

    st.markdown("""
    ### ❤️ Heart Disease Prediction System

    This application uses a machine learning pipeline
    to predict the presence of heart disease based on
    patient clinical information.

    ### Model Features

    The model uses 11 input variables:

    | Category | Features |
    |---|---|
    | Demographics | Age, Sex, Dataset |
    | Clinical | CP, Trestbps, Chol, FBS, Restecg |
    | Exercise Test | Thalch, Exang, Oldpeak |

    ### Prediction Output

    - **0:** No heart disease detected by the model.
    - **1:** Heart disease detected by the model.

    ### Technology Stack

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    """)
