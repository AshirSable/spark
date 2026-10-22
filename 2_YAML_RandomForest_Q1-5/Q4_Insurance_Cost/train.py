import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the YAML configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 2. Load the dataset (Excel file)
df = pd.read_excel(config["data"]["file"])
print("Dataset shape:", df.shape)
print(df.head())

# 3. Preprocessing
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()                       # remove repeated rows
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())      # numbers: median
    else:
        df[col] = df[col].fillna(df[col].mode()[0])     # text: most common value

# Encode the categorical column: text -> numbers
df["Smoker"] = df["Smoker"].map({"No": 0, "Yes": 1})

# 4. Separate features (X) and target (y)
target = config["data"]["target"]
X = df.drop(columns=[target])
y = df[target]

# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["model"]["random_state"]
)

# 6. Train the Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["model"]["random_state"]
)
model.fit(X_train, y_train)

# 7. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\nMean Absolute Error (MAE):", round(mae, 2))
print("R2 Score:", round(r2, 4))

# 8. Save the model as a pickle file
joblib.dump(model, config["output"]["model_file"])
print("Model saved as", config["output"]["model_file"])

# 9. Load the saved model and predict for the new record
loaded_model = joblib.load(config["output"]["model_file"])

new_data = pd.DataFrame({
    "Age": [35],
    "BMI": [27.5],
    "Children": [2],
    "Smoker": [0],            # No
    "Exercise_Hours_Per_Week": [4],
})
new_data = new_data[X.columns]   # same column order as training

prediction = loaded_model.predict(new_data)
print("\nPredicted insurance charges:", round(prediction[0], 2))
