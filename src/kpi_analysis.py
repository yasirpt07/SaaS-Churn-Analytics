import pandas as pd

df = pd.read_excel("../data/Telco_customer_churn.xlsx")

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Only drop rows where Total Charges is missing
df = df.dropna(subset=["Total Charges"])

# KPIs
total_revenue = df["Total Charges"].sum()
mrr = df["Monthly Charges"].sum()
arpu = df["Monthly Charges"].mean()
churn_rate = df["Churn"].mean() * 100

print("Total Revenue:", total_revenue)
print("Monthly Recurring Revenue:", mrr)
print("ARPU:", arpu)
print("Churn Rate:", churn_rate)