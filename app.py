"""Streamlit web app.  Run:  streamlit run app.py"""
import streamlit as st
from predict import predict

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate the annual insurance charges.")

c1, c2 = st.columns(2)
age = c1.slider("Age", 18, 64, 30)
bmi = c2.number_input("BMI", 15.0, 55.0, 27.0, step=0.1)
sex = c1.selectbox("Sex", ["male", "female"])
children = c2.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = c1.selectbox("Smoker", ["no", "yes"])
region = c2.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict cost", type="primary"):
    cost = predict(age, sex, bmi, children, smoker, region)
    st.success(f"Estimated annual cost: **${cost:,.2f}**")
    if smoker == "yes":
        st.warning("Smoking is the single biggest driver of higher premiums.")
