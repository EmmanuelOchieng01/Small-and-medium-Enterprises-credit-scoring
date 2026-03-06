import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# ----------------------------
# Load the model
# ----------------------------
model_path = Path("models") / "credit_scoring_model.pkl"
model = joblib.load(model_path)

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
    # Prepare input dataframe
    input_data = pd.DataFrame({
        "revenue": [revenue],
        "employees": [employees],
        "loan_amount": [loan_amount],
        "years_operating": [years_operating],
    })

    # Predict
    score = model.predict(input_data)[0]

    # Display results
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
