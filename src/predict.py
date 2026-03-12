import joblib
import pandas as pd

def predict_salary(input_data):
    model = joblib.load("../models/salary_model.pkl")

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)

    return prediction[0]


if __name__ == "__main__":
    sample_input = {
        "Age": 30,
        "Gender": "Male",
        "Education": "Masters",
        "Job Title": "Manager",
        "Years of Experience": 5
    }

    salary = predict_salary(sample_input)
    print("Predicted Salary:", salary)