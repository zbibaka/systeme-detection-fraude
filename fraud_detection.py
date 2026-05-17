import streamlit as st
import pandas as pd
import joblib

# Rest of your code...

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection Prediction App") 
st.markdown("Enter transaction details to predict if it's fraudulent or not.")
st.divider()
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)",min_value=0.0, value=10000.0)
newbalanceOrg = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
     input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
     }])
     prediction = model.predict(input_data)[0]
     if prediction == 1:
         st.error("The transaction is predicted to be FRAUDULENT.")
     else:    
         st.success("The transaction is predicted to be LEGITIMATE.")
