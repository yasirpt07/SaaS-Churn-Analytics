import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Tenure Months": 12,
    "Monthly Charges": 80,
    "Total Charges": 1000,
    "Contract": "Month-to-month",
    "Internet Service": "Fiber optic",
    "Payment Method": "Electronic check"
}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Response:", response.text)