# 📊 SaaS Revenue & Customer Churn Analytics

End-to-End Data Analytics & Machine Learning project simulating a SaaS subscription-based business.

This project combines **Business Intelligence, Exploratory Data Analysis (EDA), and Machine Learning** to identify churn drivers and predict high-risk customers.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges for subscription-based companies.

This project focuses on:

- Understanding churn behavior  
- Identifying revenue drivers  
- Segmenting high-risk customers  
- Building predictive ML models  
- Providing business-ready insights  

---

## 🧠 Business Objectives

- Identify key drivers of churn  
- Analyze revenue distribution by contract type  
- Segment high-value & high-risk customers  
- Build predictive churn model  
- Improve retention strategy through data-driven insights  

---

## 🗂 Project Structure

```
SaaS_Churn_Analytics/
│
├── data/
│   ├── Telco_customer_churn.xlsx
│   ├── churn_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── kpi_analysis.py
│   ├── eda_analysis.py
│   ├── risk_analysis.py
│   ├── churn_model.py
│
├── README.md
├── .gitignore
```

---

## 📈 Exploratory Data Analysis (EDA)

### Key Findings:

- Month-to-month contracts have the highest churn rate  
- Customers with higher monthly charges churn more frequently  
- Longer tenure customers are less likely to churn  
- Fiber optic internet users show higher churn probability  

**Business Insight:**  
Contract structure and pricing model significantly influence customer retention.

---

## 📊 Key Performance Indicators (KPIs)

- Total Revenue  
- Monthly Recurring Revenue (MRR)  
- Average Revenue Per User (ARPU)  
- Churn Rate  
- Revenue by Contract Type  

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression (Baseline)

- Class imbalance handled using `class_weight="balanced"`
- Used as benchmark model

---

### 2️⃣ Random Forest (Final Model)

Improved performance by capturing non-linear relationships.

#### ✅ Final Performance:

- Accuracy: **81%**
- Recall (Churn class): **58%**
- Precision: **71%**
- F1 Score: **0.64**
- ROC-AUC Score: **0.855**
- Cross-Validation ROC-AUC: ~0.85

### 🎯 Why ROC-AUC Matters?

ROC-AUC = 0.855 indicates strong ability to distinguish churn vs non-churn customers.

---

## 🔍 Top Important Features

- Tenure Months  
- Total Charges  
- Monthly Charges  
- Contract Type (Two-Year)  
- Internet Service (Fiber Optic)  
- Dependents  

These align with real-world SaaS churn drivers.

---

## 🧩 Risk Segmentation

Customers categorized into:

- High Value Customers (Above Avg CLTV)  
- High Risk Customers (High Churn Score)  
- High Value & High Risk (Priority Retention Targets)  

---

## 💾 Model Deployment Ready

The trained model is saved as:

```
data/churn_model.pkl
```

This allows future deployment via:

- Flask API  
- Streamlit dashboard  
- Web application  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Git & GitHub  

---

## 📌 Business Impact

This project demonstrates how data can:

- Reduce customer churn  
- Improve retention strategy  
- Optimize contract design  
- Increase recurring revenue  

---

## 👨‍💻 Author

**Mohammed Yasir Arafath**  
Data Analyst | Business Intelligence Enthusiast  

---

## ⭐ Future Improvements

- Hyperparameter tuning  
- XGBoost implementation  
- Model deployment (Flask API)  
- Real-time churn prediction dashboard  

---

## 🔥 Why This Project Stands Out

- End-to-End pipeline  
- Business KPIs + ML integration  
- Model comparison  
- Class imbalance handling  
- Cross-validation  
- Deployment-ready model  
- Business-driven insights  