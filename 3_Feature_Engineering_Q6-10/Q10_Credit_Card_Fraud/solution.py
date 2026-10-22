# Q10: Feature engineering - Credit Card Fraud Detection (Python / Google Colab)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Upload the dataset (Colab). Outside Colab this part is skipped.
try:
    from google.colab import files
    uploaded = files.upload()
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ---- Load
df = pd.read_excel("Credit Card Fraud Detection.xlsx")          # <- file name
print("First five records:")
print(df.head())

# ---- Explore
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nInfo:")
df.info()
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# ---- Handle missing values: median for numbers, most common value for text
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print("\nMissing values after cleaning:", df.isnull().sum().sum())

# ---- Encode Yes/No columns as 1/0
yes_no_columns = ["International_Transaction", "Card_Present", "Fraud"]
for col in yes_no_columns:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# ---- Feature engineering: create new features
# Feature 1: High_Amount (amount above 10000)
df["High_Amount"] = np.where(df["Transaction_Amount"] > 10000, 1, 0)

# Feature 2: Night_Transaction (hour before 6 am)
df["Night_Transaction"] = np.where(df["Transaction_Hour"] < 6, 1, 0)

# Feature 3: Amount_Per_Transaction (amount divided by transactions in last 24 h (ratio))
df["Amount_Per_Transaction"] = df["Transaction_Amount"] / (df["Transactions_Last_24H"] + 0.01)

# Feature 4: Distance_Category (up to 50 km = Near, 50-500 = Medium, over 500 = Far)
df["Distance_Category"] = pd.cut(
    df["Distance_From_Home_KM"],
    bins=[0, 50, 500, np.inf],
    labels=["Near", "Medium", "Far"]
)

# ---- One-hot encoding of the category column(s)
df = pd.get_dummies(
    df,
    columns=["Distance_Category"],
    dtype=int
)

# ---- Remove the ID column
df = df.drop(columns=["Transaction_ID"])

# ---- Separate features (X) and target (y)
X = df.drop(columns=["Fraud"])
y = df["Fraud"]

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])
print("Target size:", y.shape[0])

# ---- Target distribution (1 = fraud, 0 = genuine)
print(y.value_counts())
y.value_counts().plot(kind="bar")
plt.xlabel("Fraud")
plt.ylabel("Count")
plt.title("Fraud distribution")
plt.show()

# ---- Save (and download in Colab)
df.to_csv("feature_engineered_q10.csv", index=False)
if IN_COLAB:
    files.download("feature_engineered_q10.csv")

print("\nFeature engineering completed successfully!")
