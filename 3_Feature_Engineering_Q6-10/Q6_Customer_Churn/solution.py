# Q6: Feature engineering - Customer Churn Prediction (Python / Google Colab)
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
df = pd.read_csv("customer_churn.csv")          # <- file name
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

# ---- Feature engineering: create new features
# Feature 1: Long_Tenure (tenure above 24 months)
df["Long_Tenure"] = np.where(df["Tenure_Months"] > 24, 1, 0)

# Feature 2: High_Charges (monthly charges above 1000)
df["High_Charges"] = np.where(df["Monthly_Charges"] > 1000, 1, 0)

# Feature 3: Avg_Monthly_Spend (total charges divided by tenure (ratio))
df["Avg_Monthly_Spend"] = df["Total_Charges"] / (df["Tenure_Months"] + 1)

# Feature 4: Tenure_Category (0-12 months = New, 12-36 = Regular, over 36 = Loyal)
df["Tenure_Category"] = pd.cut(
    df["Tenure_Months"],
    bins=[0, 12, 36, np.inf],
    labels=["New", "Regular", "Loyal"]
)

# ---- One-hot encoding of the category column(s)
df = pd.get_dummies(
    df,
    columns=["Contract_Type", "Tenure_Category"],
    dtype=int
)

# ---- Remove the ID column
df = df.drop(columns=["Customer_ID"])

# ---- Separate features (X) and target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])
print("Target size:", y.shape[0])

# ---- Target distribution (1 = customer left, 0 = stayed)
print(y.value_counts())
y.value_counts().plot(kind="bar")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.title("Churn distribution")
plt.show()

# ---- Save (and download in Colab)
df.to_csv("feature_engineered_q6.csv", index=False)
if IN_COLAB:
    files.download("feature_engineered_q6.csv")

print("\nFeature engineering completed successfully!")
