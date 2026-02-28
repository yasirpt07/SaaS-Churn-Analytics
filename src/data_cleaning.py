import pandas as pd

# Load dataset
df = pd.read_excel("../data/Telco_customer_churn.xlsx")

print("Dataset Loaded Successfully ✅")
print(df.head())
print(df.info())

# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Drop unnecessary columns (if they exist)
drop_cols = ["Count", "Lat Long", "Latitude", "Longitude"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert Churn Label to binary
df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

# Drop missing values
df = df.dropna()

print("Data Cleaning Completed ✅")