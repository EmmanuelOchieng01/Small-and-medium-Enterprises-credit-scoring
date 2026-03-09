import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# ----------------------------
# Load model + feature columns
# ----------------------------
model_path = Path("models") / "kenya_sme_credit_model.pkl"
features_path = Path("models") / "feature_columns.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols

model, feature_cols = load_model()

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Kenya SME Credit Scoring",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Kenya SME Credit Scoring Dashboard")
st.write("Predict credit risk for small and medium enterprises in Kenya.")

# ----------------------------
# Sidebar: user inputs
# ----------------------------
st.sidebar.header("Enter SME Information")

# Map friendly labels to feature column names
friendly_labels = {
    "revenue": "Annual Revenue (KES)",
    "annual_revenue": "Annual Revenue (KES)",
    "employees": "Number of Employees",
    "num_employees": "Number of Employees",
    "loan_amount": "Requested Loan Amount (KES)",
    "years_operating": "Years in Operation",
    "years_in_business": "Years in Operation",
    "age": "Business Age (Years)",
}

user_inputs = {}
for col in feature_cols:
    label = friendly_labels.get(col, col.replace("_", " ").title())
    user_inputs[col] = st.sidebar.number_input(label, min_value=0, value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Credit Score"):
    input_data = pd.DataFrame([user_inputs])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

    st.subheader("Credit Score Result")

    try:
        score = float(prediction)
        st.markdown(
            f"<h2 style='color:#1f77b4;'>Predicted Score: {score:.2f}</h2>",
            unsafe_allow_html=True
        )
        if score >= 80:
            st.success("Excellent credit risk ✅")
        elif score >= 50:
            st.warning("Moderate credit risk ⚠️")
        else:
            st.error("High credit risk ❌")
    except (ValueError, TypeError):
        st.markdown(
            f"<h2 style='color:#1f77b4;'>Prediction: {prediction}</h2>",
            unsafe_allow_html=True
        )

    if proba is not None:
        st.write("**Prediction confidence:**")
        classes = model.classes_
        for cls, prob in zip(classes, proba):
            st.write(f"- Class `{cls}`: {prob:.1%}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Kenya SME Credit Scoring System** 💼")
