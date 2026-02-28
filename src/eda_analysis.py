import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../data/Telco_customer_churn.xlsx")

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna(subset=["Total Charges"])
df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

# 1️⃣ Churn by Contract Type
churn_contract = df.groupby("Contract")["Churn"].mean()
print("\nChurn Rate by Contract:\n", churn_contract)

churn_contract.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.show()

# 2️⃣ Revenue by Contract
revenue_contract = df.groupby("Contract")["Monthly Charges"].sum()
print("\nRevenue by Contract:\n", revenue_contract)

revenue_contract.plot(kind="bar")
plt.title("Revenue by Contract Type")
plt.ylabel("Revenue")
plt.show()

# Churn by Payment Method
churn_payment = df.groupby("Payment Method")["Churn"].mean()
print("\nChurn Rate by Payment Method:\n", churn_payment)

churn_payment.plot(kind="bar")
plt.title("Churn Rate by Payment Method")
plt.ylabel("Churn Rate")
plt.show()

# Top 5 Churn Reasons
top_reasons = df[df["Churn"] == 1]["Churn Reason"].value_counts().head(5)
print("\nTop 5 Churn Reasons:\n", top_reasons)

top_reasons.plot(kind="bar")
plt.title("Top 5 Churn Reasons")
plt.ylabel("Count")
plt.show()

# CLTV Distribution
plt.hist(df["CLTV"], bins=30)
plt.title("Customer Lifetime Value Distribution")
plt.xlabel("CLTV")
plt.ylabel("Frequency")
plt.show()