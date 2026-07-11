# ============================================================
# MOVIE RATING PREDICTION WITH PYTHON
# CODSOFT DATA SCIENCE INTERNSHIP - TASK 2
# ============================================================

# ==========================
# Import Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

# ============================================================
# Remove Missing Values
# ============================================================

df = df.dropna()

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)

# ============================================================
# Encode Categorical Columns
# ============================================================

le = LabelEncoder()

df["Genre"] = le.fit_transform(df["Genre"])
df["Director"] = le.fit_transform(df["Director"])
df["Actor 1"] = le.fit_transform(df["Actor 1"])
df["Actor 2"] = le.fit_transform(df["Actor 2"])
df["Actor 3"] = le.fit_transform(df["Actor 3"])

# ============================================================
# Features and Target
# ============================================================

X = df[["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = df["Rating"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ============================================================
# Train Model
# ============================================================

model = RandomForestRegressor(
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# ============================================================
# Predict Ratings
# ============================================================

predictions = model.predict(X_test)

print("\nFirst 10 Predicted Ratings:")
print(predictions[:10])

# ============================================================
# Mean Absolute Error
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nMean Absolute Error (MAE):", round(mae,3))

# ============================================================
# Graph 1 : Actual vs Predicted Ratings
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    color="blue",
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted Movie Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Graph 2 : Feature Importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(8,5))

importance.sort_values().plot(
    kind="bar",
    color="teal"
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=30)

plt.grid(True)

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Graph 3 : Predicted Ratings Distribution
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    predictions,
    bins=20,
    color="orange",
    edgecolor="black"
)

plt.title("Predicted Ratings Distribution")
plt.xlabel("Predicted Rating")
plt.ylabel("Frequency")

plt.grid(True)

plt.savefig(
    "predicted_ratings_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Completed
# ============================================================

print("\n=====================================")
print("Project Completed Successfully")
print("=====================================")

print("""
Generated PNG Files

1. actual_vs_predicted.png
2. feature_importance.png
3. predicted_ratings_distribution.png

All PNG files are saved in the project folder.
""")