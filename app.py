import streamlit as st
from predict_page import show_predict
from show_explore import show_explore


page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict()
else:
    show_explore()
