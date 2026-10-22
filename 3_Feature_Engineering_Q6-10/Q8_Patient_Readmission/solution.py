# Q8: Feature engineering - Patient Readmission (Python / Google Colab)
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
df = pd.read_excel("Patient_Readmission_Data.xlsx")          # <- file name
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
yes_no_columns = ["Follow_Up_Visit", "Readmitted_Within_30_Days"]
for col in yes_no_columns:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# ---- Feature engineering: create new features
# Feature 1: Elderly (age above 60)
df["Elderly"] = np.where(df["Age"] > 60, 1, 0)

# Feature 2: Long_Stay (hospital stay above 5 days)
df["Long_Stay"] = np.where(df["Length_of_Stay_Days"] > 5, 1, 0)

# Feature 3: Meds_Per_Condition (medicines divided by chronic conditions (ratio))
df["Meds_Per_Condition"] = df["Medication_Count"] / (df["Chronic_Conditions"] + 0.01)

# Feature 4: Age_Group (up to 40 = Young, 40-60 = Middle, over 60 = Senior)
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 40, 60, np.inf],
    labels=["Young", "Middle", "Senior"]
)

# ---- One-hot encoding of the category column(s)
df = pd.get_dummies(
    df,
    columns=["Age_Group"],
    dtype=int
)

# ---- Remove the ID column
df = df.drop(columns=["Patient_ID"])

# ---- Separate features (X) and target (y)
X = df.drop(columns=["Readmitted_Within_30_Days"])
y = df["Readmitted_Within_30_Days"]

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])
print("Target size:", y.shape[0])

# ---- Target distribution (1 = readmitted within 30 days, 0 = not)
print(y.value_counts())
y.value_counts().plot(kind="bar")
plt.xlabel("Readmitted_Within_30_Days")
plt.ylabel("Count")
plt.title("Readmitted_Within_30_Days distribution")
plt.show()

# ---- Save (and download in Colab)
df.to_csv("feature_engineered_q8.csv", index=False)
if IN_COLAB:
    files.download("feature_engineered_q8.csv")

print("\nFeature engineering completed successfully!")
