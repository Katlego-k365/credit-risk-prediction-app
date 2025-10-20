#Importing Libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained Extra Trees model
model = joblib.load("extra_trees_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex","Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Provide applicant information to evaluate their likelihood of credit default")


# User Input Fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male","female"])
job = st.number_input("Job (0-3)", min_value= 0, max_value= 3, value= 1)
housing = st.selectbox("Housing", ["own","rent","free"])
saving_accounts = st.selectbox("Saving accounts", ["little","moderate","rich", "quite rich"])
checking_account = st.selectbox("Checking account", ["little","moderate","rich"])
credit_amount = st.number_input("Credit amount", min_value= 0, value=1000)
duration = st.number_input("Duration (months)", min_value= 1, value= 12)


# Encode categorical inputs using the saved label encoders
input_df = pd.DataFrame({
    "Age" : [age],
    "Sex" : [encoders["Sex"].transform([sex])[0]],
    "Job" : [job],
    "Housing" : [encoders["Housing"].transform([housing])[0]],
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount":[credit_amount],
    "Duration": [duration]
})
#Make Prediction on Button Click
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

 # Display the result
    if pred == 1:
        st.success("Credit Risk Assessment: **Low Risk**")
    else:
        st.error("Credit Risk Assessment: **Potential Default Risk**")    