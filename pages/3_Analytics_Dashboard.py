import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("📊 SaaS Churn Analytics Dashboard")
st.write("Upload customer dataset to analyze churn risk insights.")

MODEL_PATH = "data/saas_churn_pipeline.pkl"

# ==============================
# RISK COLOR MAP (PROFESSIONAL)
# ==============================
risk_colors = {
    "Low": "#00C853",       # Green
    "Medium": "#FFB300",    # Yellow-Orange
    "High": "#D50000"       # Red
}

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload CSV for Analytics", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # ==============================
    # REQUIRED COLUMNS
    # ==============================
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

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Please train model via Bulk Scoring first.")
        st.stop()

    # ==============================
    # LOAD MODEL
    # ==============================
    model = joblib.load(MODEL_PATH)

    X = df[required_columns]
    probabilities = model.predict_proba(X)[:, 1]

    df["Churn Probability"] = probabilities

    df["Risk Level"] = df["Churn Probability"].apply(
        lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
    )

    st.success("✅ Analytics Generated Successfully")

    # ==============================
    # KPI SECTION
    # ==============================
    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Avg Churn Risk %", round(df["Churn Probability"].mean() * 100, 2))
    col3.metric("High Risk Customers", (df["Risk Level"] == "High").sum())

    st.divider()

    # ==============================
    # RISK DISTRIBUTION PIE
    # ==============================
    st.subheader("📌 Risk Level Distribution")

    risk_chart = px.pie(
        df,
        names="Risk Level",
        color="Risk Level",
        color_discrete_map=risk_colors,
        title="Customer Risk Segmentation"
    )

    risk_chart.update_layout(
        paper_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(risk_chart, use_container_width=True)

    # ==============================
    # PLAN TYPE VS RISK
    # ==============================
    st.subheader("📈 Plan Type vs Average Churn Risk")

    plan_risk = df.groupby("plan_type")["Churn Probability"].mean().reset_index()

    bar_chart = px.bar(
        plan_risk,
        x="plan_type",
        y="Churn Probability",
        title="Average Risk by Plan Type",
        text_auto=True,
        color_discrete_sequence=["#42A5F5"]
    )

    bar_chart.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(bar_chart, use_container_width=True)

    # ==============================
    # USAGE VS CHURN RISK
    # ==============================
    st.subheader("📉 Usage vs Churn Risk")

    scatter = px.scatter(
        df,
        x="avg_weekly_usage_hours",
        y="Churn Probability",
        color="Risk Level",
        color_discrete_map=risk_colors,
        title="Usage Behavior vs Risk"
    )

    scatter.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(scatter, use_container_width=True)

    # ==============================
    # SUPPORT TICKETS IMPACT
    # ==============================
    st.subheader("🎫 Support Tickets Impact")

    ticket_chart = px.box(
        df,
        x="Risk Level",
        y="support_tickets",
        color="Risk Level",
        color_discrete_map=risk_colors,
        title="Support Tickets vs Risk Level"
    )

    ticket_chart.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(ticket_chart, use_container_width=True)

    # ==============================
    # DOWNLOAD ANALYTICS FILE
    # ==============================
    st.divider()

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Analytics Data",
        csv,
        "analytics_output.csv",
        "text/csv"
    )