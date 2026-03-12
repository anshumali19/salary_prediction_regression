import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("models/salary_model.pkl")

st.title("Salary Prediction Using Regression")

# ---- INPUTS ----
age = st.slider("Age", 18, 65, 25)
experience = st.slider("Years of Experience", 0, 40, 2)

gender = st.selectbox("Gender", ["Male", "Female"])

education = st.selectbox(
    "Education Level",
    ["Bachelors", "Masters", "PhD"]
)

job = st.text_input(
    "Job Title (Example: Manager, Analyst, Developer)",
    "Developer"
)

country = st.text_input("Country", "USA")

race = st.selectbox(
    "Race",
    ["White", "Black", "Asian", "Hispanic", "Other"]
)

senior = st.selectbox("Senior Level", [0, 1])

# ---- PREDICTION ----
if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job,
        "Years of Experience": experience,
        "Country": country,
        "Race": race,
        "Senior": senior
    }])

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹ {round(prediction[0], 2)}")