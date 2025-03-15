import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('xgboost_churn_model.pk1')  # Ensure correct filename

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This is a simple demo of a customer churn prediction app.")

# User input fields
tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("💰 Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
total_charges = st.number_input("💵 Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
contract = st.selectbox("📜 Contract Type", ["Month-to-Month", "One Year", "Two Year"])
payment_method = st.selectbox("💳 Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])

# Converting categorical inputs
contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
payment_map = {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}

# Dummy variables for missing categorical features
dummy_features = np.zeros(25)  # 30 total - 5 existing

# Combine all features into an array
input_features = np.concatenate([[tenure, monthly_charges, total_charges, contract_map[contract], payment_map[payment_method]], dummy_features])

# Reshape for model
input_features = input_features.reshape(1, -1)

if st.button(" Predict Churn"):
    churn_prob = model.predict_proba(input_features)[0][1]  # Probability of churn
    st.write(f"**Churn Probability:** {churn_prob:.2%}")

    # Display result
    if churn_prob > 0.5:
        st.error("⚠️ High risk of churn! Consider retention strategies.")
    else:
        st.success("✅ Low risk of churn! Customer is likely to stay.")
