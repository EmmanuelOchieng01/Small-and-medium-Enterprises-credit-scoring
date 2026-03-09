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
# Sidebar: user inputs (matched to actual CSV columns)
# ----------------------------
st.sidebar.header("Enter SME Information")

business_age           = st.sidebar.number_input("Business Age (Years)", min_value=0, value=3)
employees              = st.sidebar.number_input("Number of Employees", min_value=1, value=5)
sector                 = st.sidebar.selectbox("Sector", ["Retail", "Manufacturing", "Agriculture", "Services", "Technology", "Other"])
location               = st.sidebar.selectbox("Location", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Other"])
monthly_revenue        = st.sidebar.number_input("Monthly Revenue (KES)", min_value=0, value=50000)
monthly_expenses       = st.sidebar.number_input("Monthly Expenses (KES)", min_value=0, value=30000)
profit_margin          = st.sidebar.slider("Profit Margin (%)", 0.0, 100.0, 20.0)
avg_account_balance    = st.sidebar.number_input("Avg Bank Account Balance (KES)", min_value=0, value=20000)
transaction_frequency  = st.sidebar.number_input("Monthly Transaction Frequency", min_value=0, value=10)
loan_repayment_history = st.sidebar.slider("Loan Repayment History Score (0-10)", 0, 10, 5)
existing_loans         = st.sidebar.number_input("Number of Existing Loans", min_value=0, value=1)
collateral_value       = st.sidebar.number_input("Collateral Value (KES)", min_value=0, value=100000)

# ----------------------------
# Encode categoricals the same way training did
# ----------------------------
from sklearn.preprocessing import LabelEncoder

sector_map   = {"Retail": 0, "Manufacturing": 1, "Agriculture": 2, "Services": 3, "Technology": 4, "Other": 5}
location_map = {"Nairobi": 0, "Mombasa": 1, "Kisumu": 2, "Nakuru": 3, "Eldoret": 4, "Other": 5}

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Credit Risk"):
    input_data = pd.DataFrame([{
        "business_age":           business_age,
        "employees":              employees,
        "sector":                 sector_map.get(sector, 5),
        "location":               location_map.get(location, 5),
        "monthly_revenue":        monthly_revenue,
        "monthly_expenses":       monthly_expenses,
        "profit_margin":          profit_margin,
        "avg_account_balance":    avg_account_balance,
        "transaction_frequency":  transaction_frequency,
        "loan_repayment_history": loan_repayment_history,
        "existing_loans":         existing_loans,
        "collateral_value":       collateral_value,
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

    st.subheader("Credit Risk Result")

    if prediction == 1 or str(prediction).lower() in ["default", "high", "yes", "1"]:
        st.error("⚠️ High Credit Risk — Likely to Default")
    else:
        st.success("✅ Low Credit Risk — Creditworthy")

    st.markdown(f"**Raw Prediction:** `{prediction}`")

    if proba is not None:
        st.write("**Prediction Confidence:**")
        classes = model.classes_
        for cls, prob in zip(classes, proba):
            label = "Default" if str(cls) in ["1", "yes", "default"] else "No Default"
            st.progress(float(prob), text=f"{label}: {prob:.1%}")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.dataframe(input_data)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Kenya SME Credit Scoring System** 💼")
