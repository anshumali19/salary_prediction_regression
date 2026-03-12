# Name : Anshumali Sharma

# Salary Prediction Using Regression

## Overview
This project builds a machine learning model to predict employee salaries using regression techniques. The model analyzes several factors such as age, education level, job title, years of experience, country, and other demographic information to estimate the expected salary.

The project demonstrates a complete **end-to-end machine learning pipeline**, including data preprocessing, model training, evaluation, model saving, and deployment through a web application.

A **Streamlit web interface** allows users to input their details and receive a predicted salary instantly.

---

# Problem Statement
Salary determination depends on multiple factors including experience, education, role, and location. Companies often analyze historical data to determine salary ranges.

The objective of this project is to build a **machine learning regression model** capable of predicting salary based on these influencing factors.

---

# Dataset
The dataset used in this project was downloaded from Kaggle and contains employee information used to train the model.

After preprocessing and cleaning, the dataset contains:

- **6684 records**
- **9 features**

## Dataset Features

| Feature | Description |
|------|------|
| Age | Age of the employee |
| Gender | Gender of the employee |
| Education Level | Highest education qualification |
| Job Title | Employee job position |
| Years of Experience | Total professional experience |
| Country | Country where the employee works |
| Race | Demographic race category |
| Senior | Indicates if the employee is in a senior position |
| Salary | Target variable (salary to be predicted) |

The **Salary column** is the target variable that the model learns to predict.

---

# Project Workflow

The system follows a typical machine learning workflow:
Dataset
↓
Data Cleaning
↓
Feature Encoding & Scaling
↓
Train-Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Saving
↓
Streamlit Web App
↓
User Salary Prediction


---

# Data Preprocessing

Real-world datasets often contain missing or inconsistent values. Before training the model, the dataset must be cleaned.

### Steps Performed

### 1. Load Dataset
The dataset is loaded using the **Pandas** library.

```python
df = pd.read_csv(filepath)



2. Clean Salary Column

The salary column is converted to numeric values.

df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")1


Invalid values are converted to NaN.

3. Remove Missing Data

Rows containing missing values are removed.

df = df.dropna()
This ensures the model trains only on valid data.

```
### Feature Engineering

The dataset contains two types of features: numeric and categorical.

Numeric Features

Age
Years of Experience
Senior

These features are scaled using StandardScaler so that values are normalized before entering the model.

Categorical Features

Gender
Education Level
Job Title
Country
Race



Machine learning models cannot process text data directly, so these columns are converted to numeric format using OneHotEncoder.

Example:
Gender
Male   → [1,0]
Female → [0,1]


### Column Transformer

A ColumnTransformer is used to apply different preprocessing steps to different column types.
Numeric Features      → StandardScaler
Categorical Features  → OneHotEncoder

This ensures that every feature is processed appropriately before entering the model.

Machine Learning Model

The algorithm used in this project is Random Forest Regression.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce accurate predictions.

## Advantages

Handles nonlinear relationships

Reduces overfitting

Works well with mixed data types

High prediction accuracy

### Model Training

The dataset is split into training and testing sets.
80% Training Data
20% Testing Data

The model learns patterns from the training data and is evaluated using the testing data.

### Machine Learning Pipeline

A Pipeline is used to combine preprocessing and model training.

Pipeline structure:
Pipeline
   ├── ColumnTransformer (Preprocessing)
   └── RandomForestRegressor (Model)

## Advantages:

Cleaner architecture

Prevents data leakage

Ensures consistent preprocessing during prediction

### Model Evaluation

Two evaluation metrics are used.

Mean Absolute Error (MAE)

MAE measures the average difference between predicted and actual salary values.

MAE ≈ 3844

This means the predicted salary differs from the actual salary by about 3844 units on average.

R² Score (Coefficient of Determination)

R² measures how well the model explains variation in salary.

## Range:
0 → poor model
1 → perfect model

### Project result:
R² Score = 0.97

This means the model explains 97% of the variation in salary, indicating very strong performance.

## Model Saving

The trained model is saved using Joblib so that it can be reused without retraining.

joblib.dump(model, "salary_model.pkl")

The saved model is stored inside the models/ folder.

### Streamlit Web Application

A web interface is built using Streamlit so users can interact with the model.

The application allows users to enter:
 ```
Age

Gender

Education Level

Job Title

Years of Experience

Country

Race

Senior Level
```
After clicking Predict Salary, the app sends the input to the trained model and displays the predicted salary.

### Example Prediction

Example Input:
Age: 30
Experience: 6 years
Education: Bachelors
Job Title: Manager
Country: USA

Predicted Output:
Predicted Salary: ₹101000

The prediction is based on patterns learned from 6684 dataset records.


### Technologies Used


Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical computations
Scikit-learn	Machine learning algorithms
Joblib	Model serialization
Streamlit	Web application interface
Kaggle	Dataset source


### Project Structure

```
salary_prediction_regression/
│
├── data/
│   ├── Salary.csv
│
├── models/
│   └── salary_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```


## How to Run the Project

1. Install dependencies
pip install -r requirements.txt

2. Train the model
cd src
python train.py

3. Run the Streamlit app
streamlit run app.py

Then open the browser at:

http://localhost:8501


### Future Improvements

Possible enhancements include:

Hyperparameter tuning using GridSearchCV

Feature importance visualization

Comparing multiple regression models

Using larger datasets

Deploying the application to cloud platforms

### Conclusion

This project demonstrates how machine learning can be used to predict salaries based on multiple professional and demographic features.

By combining data preprocessing, feature encoding, and Random Forest Regression, the model achieves strong predictive performance.

The final system allows users to interact with the model through a web interface and obtain salary predictions instantly.

