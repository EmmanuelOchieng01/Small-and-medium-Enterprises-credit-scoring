import streamlit as st

st.title("Kenya SME Credit Scoring")

revenue = st.number_input("Annual Revenue")
employees = st.number_input("Number of Employees")
years = st.number_input("Years in Business")

if st.button("Calculate Score"):
    score = (revenue * 0.3) + (employees * 0.2) + (years * 0.5)
    st.write("Estimated Credit Score:", score)
