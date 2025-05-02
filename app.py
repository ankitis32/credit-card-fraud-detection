# Fraud Detection App with separate amount and time scalers
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scalers
try:
    model = joblib.load('C:/Users/a.tiwari/OneDrive - Solutions+/Desktop/Programming/Python/AI/credit-card-fraud-detection/xgboost_model.pkl')
    amount_scaler = joblib.load('C:/Users/a.tiwari/OneDrive - Solutions+/Desktop/Programming/Python/AI/credit-card-fraud-detection/amount_scaler.pkl')
    time_scaler = joblib.load('C:/Users/a.tiwari/OneDrive - Solutions+/Desktop/Programming/Python/AI/credit-card-fraud-detection/time_scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")

st.markdown("""
This app uses a trained XGBoost model to detect potential credit card fraud based on transaction features.
""")

# Sidebar for user input
st.sidebar.header("Input Transaction Details")

def user_input():
    input_data = {}
    input_data["Time"] = st.sidebar.slider("Time (seconds)", 0.0, 172800.0, 10000.0)  # Up to 48 hours
    for i in range(1, 29):
        input_data[f"V{i}"] = st.sidebar.slider(f"V{i}", -30.0, 30.0, 0.0)
    input_data["Amount"] = st.sidebar.slider("Amount ($)", 0.0, 2500.0, 100.0)
    return pd.DataFrame([input_data])

input_df = user_input()

# --- Manual Prediction Button ---
if st.button("🔍 Predict Transaction"):
    try:
        # Scale 'Amount' and 'Time'
        input_df["scaled_amount"] = amount_scaler.transform(input_df[["Amount"]].values)
        input_df["scaled_time"] = time_scaler.transform(input_df[["Time"]].values)

        # Drop original columns
        input_df.drop(["Amount", "Time"], axis=1, inplace=True)

        # Reorder columns to match training data
        input_df = input_df[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected!\n\nProbability of fraud: {proba:.2%}")
        else:
            st.success(f"✅ Legitimate Transaction\n\nProbability of fraud: {proba:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Batch Prediction Section ---
st.markdown("---")
st.subheader("🔍 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch fraud detection", type=["csv"])

if uploaded_file is not None:
    if st.button("⚙️ Run Batch Prediction"):
        try:
            batch_data = pd.read_csv(uploaded_file)

            # Ensure required columns are present
            required_cols = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
            if not all(col in batch_data.columns for col in required_cols):
                st.error(f"CSV must contain the following columns: {required_cols}")
                st.stop()

            # Drop 'Class' if present
            if 'Class' in batch_data.columns:
                batch_data.drop('Class', axis=1, inplace=True)

            # Scale Amount and Time
            batch_data["scaled_amount"] = amount_scaler.transform(batch_data[["Amount"]].values)
            batch_data["scaled_time"] = time_scaler.transform(batch_data[["Time"]].values)

            # Drop original columns
            batch_data.drop(["Amount", "Time"], axis=1, inplace=True)

            # Reorder columns
            batch_data = batch_data[model.feature_names_in_]

            # Predict
            batch_preds = model.predict(batch_data)
            batch_proba = model.predict_proba(batch_data)[:, 1]

            batch_data["Fraud_Prediction"] = batch_preds
            batch_data["Fraud_Probability"] = batch_proba

            # Display results
            st.write("### Results:")
            st.dataframe(batch_data)

            st.download_button(
                label="Download Results as CSV",
                data=batch_data.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")