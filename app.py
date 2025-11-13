import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('best_model.pkl')

st.title("Student Exam Score Predictor")

study_hours = st.slider("Study hours per Day", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Level (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

part_time_job_enoceded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_job_enoceded]])
    predicted_score = model.predict(input_data)[0]

    predicted_score = max(0, min(100, predicted_score))
    st.success(f"The predicted exam score is: {predicted_score:.2f}")