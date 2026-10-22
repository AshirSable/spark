# Q7: Feature engineering - House Price Prediction (Python / Google Colab)
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
df = pd.read_excel("house_Price_Data.xlsx")          # <- file name
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
# Feature 1: Total_Rooms (bedrooms plus bathrooms)
df["Total_Rooms"] = df["Bedrooms"] + df["Bathrooms"]

# Feature 2: Area_Per_Bedroom (area divided by bedrooms (ratio))
df["Area_Per_Bedroom"] = df["Area_SqFt"] / (df["Bedrooms"] + 0.01)

# Feature 3: Large_House (area above 1500 sq ft)
df["Large_House"] = np.where(df["Area_SqFt"] > 1500, 1, 0)

# Feature 4: Building_Age_Category (0-5 years = New, 5-10 = Medium, over 10 = Old)
df["Building_Age_Category"] = pd.cut(
    df["Building_Age"],
    bins=[0, 5, 10, np.inf],
    labels=["New", "Medium", "Old"]
)

# ---- One-hot encoding of the category column(s)
df = pd.get_dummies(
    df,
    columns=["Building_Age_Category"],
    dtype=int
)

# ---- Remove the ID column
df = df.drop(columns=["House_ID"])

# ---- Separate features (X) and target (y)
X = df.drop(columns=["House_Price_INR"])
y = df["House_Price_INR"]

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])
print("Target size:", y.shape[0])

# ---- Target distribution (target is a price (a number), so a histogram is shown)
print(y.describe())
y.plot(kind="hist", bins=10)
plt.xlabel("House_Price_INR")
plt.ylabel("Frequency")
plt.title("House_Price_INR distribution")
plt.show()

# ---- Save (and download in Colab)
df.to_csv("feature_engineered_q7.csv", index=False)
if IN_COLAB:
    files.download("feature_engineered_q7.csv")

print("\nFeature engineering completed successfully!")
