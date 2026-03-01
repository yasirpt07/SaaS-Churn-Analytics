import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# =====================
# Load Data
# =====================

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

# =====================
# Split Features
# =====================

selected_features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
    "Internet Service",
    "Payment Method"
]

X = df[selected_features].copy()
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Identify Columns
# =====================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    X[col] = X[col].astype(str)
# =====================
# Preprocessing
# =====================

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# =====================
# Create Pipeline
# =====================

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# =====================
# Train
# =====================

pipeline.fit(X_train, y_train)

# =====================
# Evaluate
# =====================

probs = pipeline.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, probs)

print("Pipeline ROC-AUC:", roc_score)

# =====================
# Save Full Pipeline
# =====================

joblib.dump(pipeline, "../data/churn_pipeline.pkl")
print("Pipeline saved successfully!")