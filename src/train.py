import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import load_and_preprocess

def train_model():
    X, y, preprocessor = load_and_preprocess("../data/Salary.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Full pipeline (Preprocessing + Model)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/salary_model.pkl")

    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()