import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("Students_LifeStyle2.csv")

df["Performance_Level"] = pd.cut(
    df["CGPA"],
    bins=[-1, 2.0, 3.0, 4.1],
    labels=["Low", "Medium", "High"]
)

X = df.drop(columns=["Performance_Level", "CGPA", "Student_ID"], errors="ignore")
y = df["Performance_Level"]

num_cols = ["Age","Sleep_Duration","Study_Hours","Social_Media_Hours","Physical_Activity","Stress_Level"]
cat_cols = ["Gender","Department"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

clf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", clf),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(pipeline, "model.pkl")
print("✅ model.pkl (Performance Low/Medium/High) sauvegardé")
