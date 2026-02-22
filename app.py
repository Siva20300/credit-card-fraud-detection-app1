import streamlit as st
import pandas as pd
import joblib

# Load trained objects
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details")

# ---- USER INPUTS (ONLY REALISTIC FIELDS) ----
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0)

merchant_type = st.selectbox(
    "Merchant Type",
    ["grocery", "fuel", "online", "travel", "entertainment"]
)

device_type = st.selectbox(
    "Device Type",
    ["mobile", "web", "pos"]
)

# ---- AUTO-FILL PCA FEATURES (V1–V28) ----
pca_features = {f"v{i}": 0.0 for i in range(1, 29)}

# ---- PREDICTION ----
if st.button("Predict Fraud"):
    data = {
        "time": time,
        "amount": amount,
        "merchant_type": merchant_type,
        "device_type": device_type,
        **pca_features
    }

    df = pd.DataFrame([data])
    df.columns = df.columns.str.lower()

    df = pd.get_dummies(
        df,
        columns=["merchant_type", "device_type"],
        drop_first=True
    )

    df["amount"] = scaler.transform(df[["amount"]])
    df = df.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    if prob >= 0.3:
        st.error(f"🚨 Fraud Detected (Probability: {prob:.3f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability: {prob:.3f})")
