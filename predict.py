import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model():
    with open("model.pkl", "rb") as file:
        data = pickle.load(file)
    print("Model is loaded.")
    return data

data = load_model()
model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 

mapping_dictionary = dict(zip(
            [i for i in range(7)], ['Normal Weight', 'Overweight_Level_I', 'Overweight_Level_II',
       'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
       'Obesity_Type_III']
        ))

def predict_page():

    st.title("Welcome to Obesity Risk Assessment Page")
    st.write("#### Enter all necessary information to get the results")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.text_input("Age", "0")
    height = st.text_input("Height (cm)", "0")
    weight = st.text_input("Weight (kg)", "0")
    family = st.radio("Are there any people with obesity in your family?", ["yes", "no"])
    favc = st.radio("Do you consume fast food on a daily basis?", ["yes", "no"])
    fcvc = st.selectbox("How often do you consume vegetables during a day?", [0, 1, 2, 3])
    ncp = st.selectbox("Number of main meals", [0, 1, 2, 3])
    smoke = st.radio("Do you smoke?", ["yes", "no"])
    alco = st.radio("Do you drink alcohol?", ["yes", "no"])
    water = st.selectbox("Water consumption (in litres)", [0, 1, 2, 3])
    snack = st.selectbox("How often do you eat between main meals?", ["Never", "Sometimes", "Frequently", "Always"])
    monitoring = st.radio("Do you monitor calorie intake?", ["yes", "no"])
    jim = st.selectbox("Number of physical trainings a week", [0, 1, 2, 3])
    screen = st.selectbox("Screen time (hours)", [0, 1, 2, 3])
    transport = st.selectbox("What kind of transport do you use mostly?", ["Public", "Automobile", "Bike", "Walking", "Motorbike"])


    @st.experimental_dialog("Results")
    def res(pred):
        st.write(f"{mapping_dictionary[pred]}")

    btn = st.button("Assess Risk")  

    if btn:
        data = {'Gender': gender,
                'Age': int(age),
                'Height': int(height),
                'Weight': int(weight),
                'family_history_with_overweight': family,
                'FAVC': favc,
                'FCVC': int(fcvc),
                'NCP': int(ncp),
                'CAEC': snack,
                'SMOKE': smoke,
                'CH2O': int(water),
                'SCC': monitoring,
                'FAF': int(jim),
                'TUE': int(screen),
                'CALC': alco,
                'MTRANS': transport
                }
        df = pd.DataFrame([list(data.values())], columns=[list(data.keys())])
        for col in df.columns:
            if col in df.select_dtypes("object").columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])             
        prediction = model.predict(df)
        
        st.success(f"{mapping_dictionary[int(prediction)]}")
        
        lst = ['Normal Weight', 'Overweight_Level_I', 'Overweight_Level_II',
       'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
       'Obesity_Type_III']
        prob = [round(percent * 100, 2) for percent in list(model.predict_proba(df))[0]]
        st.write("Chances to get one of the following:")
        for key, val in dict(zip(lst, prob)).items():
            st.write(f"{key} - {val} %")
        


