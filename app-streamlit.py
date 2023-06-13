import pandas as pd
import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
def predict_cust_satisfaction(age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes):
    input_data = [[age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes]]
    scaler = joblib.load("output/scaler.pkl")
    scaled_input_data = scaler.transform(input_data)
    
    classifier = joblib.load("output/xgbclassifier.pkl")
    prediction = classifier.predict(scaled_input_data)
    return prediction  

def Input_Output():
    st.markdown("<h1 style='text-align: center'>Airline Passengers Satisfaction Prediction", unsafe_allow_html=True)
    st.image("./static/images/pesawat.png", width=600)
    st.markdown("<h3 style='text-align: center'>You are using Streamlit...",unsafe_allow_html=True)
    age  = st.text_input("**Enter Age**" )
    gender  = st.text_input("**Enter Gender** (0: Female, 1: Male)")
    customer_type  = st.text_input("**Enter Customer Type** (0: Loyal Customer, 1: Disloyal Customer)")
    customer_class  = st.text_input("**Enter Class** (0: Bussiness, 1: Eco, 2: Eco Plus)")
    type_of_travel  = st.text_input("**Enter Type of Travel** (0: Bussiness Travel, 1: Personal Travel)")
    checkin_service  = st.text_input("**Enter Checkin Service** (Rating scale: 1 - 5, the higher rating is the best)")
    flight_distance  = st.text_input("**Enter Flight Distance**")
    departure_delay_in_minutes  = st.text_input("**Enter Departure Delay in Minutes**")
    result = ""
    if st.button("**Click here to Predict**"):
        result = predict_cust_satisfaction(age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes)
        st.balloons() 
    # if st.button("Click here to Predict"):
    #     result = predict_cust_satisfaction(age, gender, customer_type, customer_class, type_of_travel, checkin_service, flight_distance, departure_delay_in_minutes)
    #     prediction_label = "Satisfied" if result[0] == 1 else "Dissatisfied"
    #     st.write(f"The output is {prediction_label}")
    #     st.balloons()   
    st.success('The output is {}'.format(result))
    st.markdown("**Note:**")
    st.markdown("0: Dissatisfied")
    st.markdown("1: Satisfied")

if __name__ ==  '__main__':
    Input_Output()
