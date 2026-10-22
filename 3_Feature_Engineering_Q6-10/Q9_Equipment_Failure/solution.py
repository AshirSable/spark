# Q9: Feature engineering - Manufacturing Equipment Failure (Python / Google Colab)
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
df = pd.read_excel("Equipment_Failure_Data.xlsx")          # <- file name
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
yes_no_columns = ["Failure"]
for col in yes_no_columns:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# ---- Feature engineering: create new features
# Feature 1: Overheating (temperature above 80 C)
df["Overheating"] = np.where(df["Temperature_C"] > 80, 1, 0)

# Feature 2: High_Vibration (vibration above 6 mm/s)
df["High_Vibration"] = np.where(df["Vibration_MM_S"] > 6, 1, 0)

# Feature 3: Pressure_Temp_Ratio (pressure divided by temperature (ratio))
df["Pressure_Temp_Ratio"] = df["Pressure_Bar"] / (df["Temperature_C"] + 0.01)

# Feature 4: Usage_Category (0-1000 h = Low, 1000-2000 = Medium, over 2000 = High)
df["Usage_Category"] = pd.cut(
    df["Operating_Hours"],
    bins=[0, 1000, 2000, np.inf],
    labels=["Low", "Medium", "High"]
)

# ---- One-hot encoding of the category column(s)
df = pd.get_dummies(
    df,
    columns=["Usage_Category"],
    dtype=int
)

# ---- Remove the ID column
df = df.drop(columns=["Machine_ID"])

# ---- Separate features (X) and target (y)
X = df.drop(columns=["Failure"])
y = df["Failure"]

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])
print("Target size:", y.shape[0])

# ---- Target distribution (1 = machine failed, 0 = did not)
print(y.value_counts())
y.value_counts().plot(kind="bar")
plt.xlabel("Failure")
plt.ylabel("Count")
plt.title("Failure distribution")
plt.show()

# ---- Save (and download in Colab)
df.to_csv("feature_engineered_q9.csv", index=False)
if IN_COLAB:
    files.download("feature_engineered_q9.csv")

print("\nFeature engineering completed successfully!")
