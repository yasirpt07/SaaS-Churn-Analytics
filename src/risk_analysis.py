import pandas as pd

df = pd.read_excel("../data/Telco_customer_churn.xlsx")

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna(subset=["Total Charges"])
df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

# High Value Customers (Above average CLTV)
avg_cltv = df["CLTV"].mean()
high_value = df[df["CLTV"] > avg_cltv]

# High Risk Customers (High Churn Score)
high_risk = df[df["Churn Score"] > 80]

# Intersection
high_value_risk = high_value[high_value["Churn Score"] > 80]

print("Total Customers:", len(df))
print("High Value Customers:", len(high_value))
print("High Risk Customers:", len(high_risk))
print("High Value & High Risk Customers:", len(high_value_risk))