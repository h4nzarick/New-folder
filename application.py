import streamlit as st
from predict import predict_page

page = st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore"))

if page == "Predict":
    predict_page()
else:
    pass