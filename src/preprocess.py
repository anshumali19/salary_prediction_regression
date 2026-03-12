import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_and_preprocess(filepath):

    df = pd.read_csv(filepath)

    # ---- CLEAN TARGET ----
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"])
    df = df.dropna()

    # ---- SPLIT ----
    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    # ---- FEATURE TYPES ----
    numeric_features = [
        "Age",
        "Years of Experience",
        "Senior"
    ]

    categorical_features = [
        "Gender",
        "Education Level",
        "Job Title",
        "Country",
        "Race"
    ]

    # ---- PREPROCESSING PIPELINE ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    print("Dataset shape after cleaning:", df.shape)

    return X, y, preprocessor