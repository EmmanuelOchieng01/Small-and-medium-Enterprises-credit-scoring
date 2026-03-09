import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import numpy as np

# ----------------------------
# Load the model
# ----------------------------
model_path = Path("models") / "kenya_sme_credit_model.pkl"

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError:
        # Fallback: numpy version mismatch (numpy._core vs numpy.core)
        # Re-save the model with the current numpy version to fix permanently
        import pickle
        import importlib, sys

        # Patch missing numpy._core to point to numpy.core
        if "numpy._core" not in sys.modules:
            import numpy.core as _np_core
            sys.modules["numpy._core"] = _np_core
            sys.modules["numpy._core.multiarray"] = _np_core.multiarray

        with open(path, "rb") as f:
            model = pickle.load(f)

        # Re-save with current numpy so next load works normally
        joblib.dump(model, path)
        st.toast("✅ Model re-saved for your numpy version. Restart the app once.", icon="🔧")
        return model

model = load_model(model_path)

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

revenue = st.sidebar.number_input("Annual Revenue (KES)", min_value=0, value=100000)
employees = st.sidebar.number_input("Number of Employees", min_value=1, value=5)
loan_amount = st.sidebar.number_input("Requested Loan Amount (KES)", min_value=0, value=50000)
years_operating = st.sidebar.slider("Years in Operation", 0, 50, 3)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Credit Score"):
    input_data = pd.DataFrame({
        "revenue": [revenue],
        "employees": [employees],
        "loan_amount": [loan_amount],
        "years_operating": [years_operating],
    })

    score = model.predict(input_data)[0]

    st.subheader("Credit Score Result")
    st.markdown(
        f"<h2 style='color:#1f77b4;'>Predicted Credit Score: {score:.2f}</h2>",
        unsafe_allow_html=True
    )

    if score >= 80:
        st.success("Excellent credit risk ✅")
    elif score >= 50:
        st.warning("Moderate credit risk ⚠️")
    else:
        st.error("High credit risk ❌")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Kenya SME Credit Scoring System** 💼")
