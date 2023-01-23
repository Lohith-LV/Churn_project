# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import all the app dependencies
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import shap
import matplotlib
from IPython import get_ipython
from PIL import Image

model = joblib.load("churn_model.joblib")
label_enc = joblib.load("label_encoder.joblib")
stand_scal = joblib.load("stand_scalar.joblib")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Customer Churn Prediction",
                page_icon="🚧", layout="wide")

options_gender = ['Male','Female']
options_senior_citizen = [0,1]
options_partner = ['Yes','No']
options_dependents = ['Yes','No']
options_phone_service = ['Yes','No']
options_multiple_lines = ['No phone service', 'No', 'Yes']
options_internet_service = ['DSL', 'Fiber optic', 'No']
options_online_security = ['No', 'Yes', 'No internet service']
options_online_backup = ['Yes', 'No', 'No internet service']
options_device_protection = ['Yes', 'No', 'No internet service']
options_tech_support = ['Yes', 'No', 'No internet service']
options_streaming_tv = ['Yes', 'No', 'No internet service']
options_streaming_movies = ['Yes', 'No', 'No internet service']
options_contract = ['Month-to-month', 'One year', 'Two year']
options_paperless_bill = ['Yes', 'No']
options_payment_method = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)']

def main():
    with st.form("Churn Form"):
        st.subheader("Please enter the following inputs:")
        gender = st.selectbox("Gender :",options = options_gender)
        sen_citizen =  st.selectbox("Senior Citizen :", options = options_senior_citizen)
        partner = st.selectbox("Partner :",options = options_partner)
        dependants = st.selectbox("Dependants :", options = options_dependents)
        tenure = st.number_input("Give the Tenure :")
        phone_serv = st.selectbox("Phone Service :",options = options_phone_service)
        multiple_lines = st.selectbox("Multiple Lines :",options = options_multiple_lines)
        internet_serv = st.selectbox("Internet Services :",options = options_internet_service)
        online_secu = st.selectbox("Online Services :",options = options_online_security)
        online_backup = st.selectbox("Online Backup :",options = options_online_backup)
        dev_protection = st.selectbox("Device Protection :",options = options_device_protection)
        tech_supp = st.selectbox("Tech Support :",options = options_tech_support)
        streaming_tv = st.selectbox("Streaming TV :",options = options_streaming_movies)
        streaming_mov = st.selectbox("Streaming Movies :",options = options_streaming_movies)
        contract = st.selectbox("Contract :",options = options_contract)
        paperless_billing = st.selectbox("Paperless Billing :",options = options_paperless_bill)
        payment_method = st.selectbox("Payment Method :",options = options_payment_method)
        monthly_charges = st.number_input("Monthly Charges :")
        total_charges = st.number_input("Total Charges")
        
        submit = st.form_submit_button("Predict")

    if submit:
        input_array = np.array([gender,sen_citizen,partner,dependants,phone_serv,
                                multiple_lines,internet_serv,online_secu,online_backup,
                                dev_protection,tech_supp,streaming_tv,streaming_mov,
                                contract,paperless_billing,payment_method,monthly_charges,total_charges,tenure], ndmin=2)
        #df = pd.DataFrame(input_array,columns=['gender','SeniorCitizen', 'Partner', 'Dependents','PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'tenure'])
        df = pd.DataFrame({'gender':[gender],'SeniorCitizen':[sen_citizen],'Partner':[partner],'Dependents':[dependants],'PhoneService':[phone_serv],'MultipleLines':[multiple_lines],'InternetService':[internet_serv],'OnlineSecurity':[online_secu],'OnlineBackup':[online_backup],'DeviceProtection':[dev_protection],'TechSupport':[tech_supp], 'StreamingTV':[streaming_tv],'StreamingMovies':[streaming_mov],'Contract' :[contract], 'PaperlessBilling':[paperless_billing],'PaymentMethod':[payment_method],'MonthlyCharges' :[monthly_charges],'TotalCharges':[total_charges],'tenure':[tenure]})
        for column in df.columns:
            st.write(column)
            if df[column].dtype == np.number:
                continue
            st.write(column)
            df[column] = label_enc.transform(df[column])
        
        scal_df = stand_scal.transform(df)
        
        prediction = model.predict(scal_df)
        
        if prediction == 1:
            st.write(f"Churn⚠")
        else:
            st.write(f"Retention")
            
if __name__ == '__main__':
   main()
        
print("Hello World")
