import streamlit as st
import pandas as pd
import pickle  # or joblib if you saved your model that way
from pathlib import Path

# ----------------------------
# Load model
# ----------------------------
model_path = Path("models") / "your_model.pkl"  # Replace with actual saved model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Kenya SME Credit Scoring",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Enter SME Information")

# Example features – replace/add based on your model
revenue = st.sidebar.number_input("Annual Revenue (KES)", min_value=0, value=100000)
employees = st.sidebar.number_input("Number of Employees", min_value=1, value=5)
loan_amount = st.sidebar.number_input("Requested Loan Amount (KES)", min_value=0, value=50000)
years_operating = st.sidebar.slider("Years in Operation", min_value=0, max_value=50, value=3)

# You can add more features here according to your model

# ----------------------------
# Predict button
# ----------------------------
if st.sidebar.button("Predict Credit Score"):
    # Create DataFrame for the model
    input_data = pd.DataFrame({
        "revenue": [revenue],
        "employees": [employees],
        "loan_amount": [loan_amount],
        "years_operating": [years_operating],
    })

    # Predict using the loaded model
    score = model.predict(input_data)[0]

    # ----------------------------
    # Display results
    # ----------------------------
    st.subheader("Credit Scoring Result")
    st.markdown(
        f"<h2 style='color: #1f77b4;'>Predicted Credit Score: {score:.2f}</h2>",
        unsafe_allow_html=True
    )

    # Optional color coding
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
st.markdown("**Kenya SME Credit Scoring Dashboard**  💼")
