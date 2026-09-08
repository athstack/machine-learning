"""
Streamlit app for Retail Customer Churn Prediction.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("models/retail_churn_pipeline.joblib")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Retail Customer Churn Prediction")
st.markdown("Enter customer features to predict churn probability.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run the notebook first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.sidebar.header("Model Info")
st.sidebar.write(f"Model type: {type(model).__name__}")
st.sidebar.write("Pipeline: Preprocessor -> Feature Selection -> Classifier")

st.subheader("Customer Features")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    total_transactions = st.number_input("Total Transactions", min_value=0, value=5)
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=500.0)
    total_quantity = st.number_input("Total Quantity", min_value=0, value=10)

with col2:
    avg_transaction_value = st.number_input("Avg Transaction Value ($)", min_value=0.0, value=100.0)
    avg_discount = st.number_input("Avg Discount", min_value=0.0, max_value=100.0, value=10.0)
    recency_days = st.number_input("Recency (days)", min_value=0, value=30)
    customer_lifetime_days = st.number_input("Customer Lifetime (days)", min_value=0, value=365)

with col3:
    total_interactions = st.number_input("Total Interactions", min_value=0, value=5)
    total_tickets = st.number_input("Total Support Tickets", min_value=0, value=1)
    avg_satisfaction = st.number_input("Avg Ticket Satisfaction", min_value=0.0, max_value=5.0, value=3.0)
    total_reviews = st.number_input("Total Reviews", min_value=0, value=2)
    avg_rating = st.number_input("Avg Rating", min_value=0.0, max_value=5.0, value=3.5)

if st.button("Predict Churn", type="primary"):
    input_data = pd.DataFrame([{
        "age": age,
        "total_transactions": total_transactions,
        "total_spend": total_spend,
        "total_quantity": total_quantity,
        "avg_transaction_value": avg_transaction_value,
        "avg_discount": avg_discount,
        "recency_days": recency_days,
        "customer_lifetime_days": customer_lifetime_days,
        "total_interactions": total_interactions,
        "total_tickets": total_tickets,
        "avg_satisfaction": avg_satisfaction,
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.divider()
        if prediction == 1:
            st.error(f"**HIGH CHURN RISK** - Probability: {probability[1]:.2%}")
        else:
            st.success(f"**LOW CHURN RISK** - Probability: {probability[1]:.2%}")

        st.subheader("Probability Breakdown")
        col_a, col_b = st.columns(2)
        col_a.metric("No Churn", f"{probability[0]:.2%}")
        col_b.metric("Churn", f"{probability[1]:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Ensure all required features match the training data schema.")
