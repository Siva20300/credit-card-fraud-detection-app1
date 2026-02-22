import streamlit as st
import pandas as pd
import joblib
import os

# Debug: show files in directory
st.write("Files in directory:", os.listdir())

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

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

st.subheader("PCA Features (V1–V28)")
pca = {}
for i in range(1, 29):
    pca[f"v{i}"] = st.number_input(f"V{i}", value=0.0)

if st.button("Predict Fraud"):
    data = {
        "time": time,
        "amount": amount,
        "merchant_type": merchant_type,
        "device_type": device_type,
        **pca
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
