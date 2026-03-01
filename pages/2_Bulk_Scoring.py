import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

st.title("📂 Bulk Customer Churn Scoring (SaaS Version)")

MODEL_PATH = "data/saas_churn_pipeline.pkl"

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

# =========================
# TRAIN MODEL FUNCTION
# =========================
def train_model(df):

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    features = [
        "tenure_months",
        "monthly_fee",
        "avg_weekly_usage_hours",
        "support_tickets",
        "payment_failures",
        "last_login_days_ago",
        "plan_type"
    ]

    X = df[features]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [
        "tenure_months",
        "monthly_fee",
        "avg_weekly_usage_hours",
        "support_tickets",
        "payment_failures",
        "last_login_days_ago"
    ]

    categorical_features = ["plan_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)

    st.success(f"✅ Model trained successfully (ROC-AUC: {round(auc,3)})")

    os.makedirs("data", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline


# =========================
# MAIN LOGIC
# =========================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Validate required columns
    required_columns = [
        "tenure_months",
        "monthly_fee",
        "avg_weekly_usage_hours",
        "support_tickets",
        "payment_failures",
        "last_login_days_ago",
        "plan_type"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    # =========================
    # Train model if not exists
    # =========================
    if not os.path.exists(MODEL_PATH):

        st.info("🔄 No trained model found. Training now...")
        model = train_model(df)

    else:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Loaded existing trained model")

    # =========================
    # Bulk Scoring
    # =========================

    X_score = df[required_columns]

    probabilities = model.predict_proba(X_score)[:,1]

    df["Churn Probability"] = probabilities

    df["Risk Level"] = df["Churn Probability"].apply(
        lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
    )

    st.subheader("📈 Scored Results")
    st.dataframe(df.head())

    # Download
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Scored CSV",
        csv,
        "scored_customers.csv",
        "text/csv"
    )