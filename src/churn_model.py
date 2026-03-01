import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# =========================
# Load and Clean Data
# =========================

df = pd.read_excel("../data/Telco_customer_churn.xlsx")

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna(subset=["Total Charges"])

df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

df = df.drop(columns=[
    "CustomerID",
    "Churn Label",
    "Churn Reason",
    "Churn Value",
    "Churn Score",
    "CLTV"
], errors="ignore")

df = pd.get_dummies(df, drop_first=True)

# =========================
# Split Data
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔥 SAVE COLUMN STRUCTURE BEFORE TRAINING
joblib.dump(X.columns.tolist(), "../data/model_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scaling
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Random Forest Model
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

roc_score = roc_auc_score(y_test, probabilities)
print("\nRandom Forest ROC-AUC:", roc_score)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("\nCross-Validation ROC-AUC Scores:", cv_scores)
print("Average CV ROC-AUC:", cv_scores.mean())

# =========================
# Save Artifacts
# =========================

joblib.dump(scaler, "../data/scaler.pkl")
joblib.dump(model, "../data/churn_model.pkl")

print("\nModel, scaler and columns saved successfully.")