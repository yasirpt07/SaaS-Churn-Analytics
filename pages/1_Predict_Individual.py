import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import io
from utils.pdf_report import generate_pdf

# =========================
# Page Setup
# =========================
st.set_page_config(layout="wide")
st.title("🔮 Individual Customer Churn Prediction")
st.write("Predict churn risk and generate AI-powered business insights.")

st.divider()

# =========================
# Load Model
# =========================
model = joblib.load("data/churn_pipeline.pkl")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 5000.0, 50.0)
total = st.sidebar.number_input("Total Charges", 0.0, 20000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

predict_btn = st.sidebar.button("🚀 Predict Churn Risk")

# =========================
# Prediction Logic
# =========================
if predict_btn:

    input_data = pd.DataFrame([{
        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "Contract": contract,
        "Internet Service": internet,
        "Payment Method": payment
    }])

    probability = model.predict_proba(input_data)[0][1]
    risk_percent = round(probability * 100, 2)

    # =========================
    # Risk Classification
    # =========================
    if risk_percent < 40:
        segment = "Loyal Customer"
        bar_color = "#00FF88"   # Green
    elif risk_percent < 70:
        segment = "At Risk"
        bar_color = "#FFA500"   # Orange
    else:
        segment = "Critical Risk"
        bar_color = "#FF4B4B"   # Red

    col1, col2 = st.columns([1,2])

    # =========================
    # KPI & PDF Section
    # =========================
    with col1:
        st.subheader("📊 Risk Score")
        st.metric("Churn Probability", f"{risk_percent}%")

        if risk_percent < 40:
            st.success("🟢 Low Risk")
        elif risk_percent < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # Generate PDF in memory
        pdf_buffer = io.BytesIO()
        generate_pdf(pdf_buffer, risk_percent, segment)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name="Churn_Report.pdf",
            mime="application/pdf"
        )

    # =========================
    # 🚀 Dynamic Gauge Chart
    # =========================
    with col2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            title={'text': "Churn Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},  # Dynamic color
                'steps': [
                    {'range': [0, 40], 'color': "#0F3D2E"},
                    {'range': [40, 70], 'color': "#3D2F0F"},
                    {'range': [70, 100], 'color': "#3D0F0F"},
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="#0E1117",
            font={'color': "white"},
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # Customer Segmentation
    # =========================
    st.subheader("📌 Customer Segmentation")

    seg_df = pd.DataFrame({
        "Segment": ["Loyal Customer", "At Risk", "Critical Risk"],
        "Value": [40, 30, 30]
    })

    seg_chart = px.pie(
        seg_df,
        names="Segment",
        values="Value",
        title=f"Customer Classified As: {segment}"
    )
    st.plotly_chart(seg_chart, use_container_width=True)

    # =========================
    # SHAP Explainability
    # =========================
    st.subheader("🧠 Model Explainability")

    try:
        rf_model = model.named_steps['model']
        transformed = model.named_steps['preprocessing'].transform(input_data)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(transformed)

        shap_df = pd.DataFrame({
            "Feature Impact": shap_values[1][0]
        })

        shap_df["Feature"] = shap_df.index
        shap_df = shap_df.sort_values("Feature Impact", ascending=False).head(6)

        shap_chart = px.bar(
            shap_df,
            x="Feature Impact",
            y="Feature",
            orientation='h',
            title="Top Drivers of Churn"
        )

        st.plotly_chart(shap_chart, use_container_width=True)

    except:
        st.info("SHAP explanation available for tree-based models.")

    # =========================
    # Business Insights
    # =========================
    st.subheader("💡 Business Insights")

    insights = []

    if contract == "Month-to-month":
        insights.append("• Month-to-month contracts increase churn risk.")
    if internet == "Fiber optic":
        insights.append("• Fiber customers historically show higher churn.")
    if monthly > 100:
        insights.append("• High monthly charges may increase churn.")
    if tenure < 12:
        insights.append("• New customers are more likely to churn.")

    if len(insights) == 0:
        insights.append("• No strong churn drivers detected.")

    for item in insights:
        st.write(item)