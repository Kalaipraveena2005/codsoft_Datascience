# ==========================================
# SALES PREDICTION USING PYTHON
# CodSoft Internship - Task 4
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("advertising.csv")

print("First 5 Rows:\n")
print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# Train Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)

print("\nFirst 10 Predicted Sales:\n")
print(predictions[:10])

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# ==========================================
# GRAPH 1 : TV vs Sales
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["TV"],
    df["Sales"],
    color="blue"
)

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("tv_vs_sales.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 2 : Radio vs Sales
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Radio"],
    df["Sales"],
    color="green"
)

plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Budget")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("radio_vs_sales.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 3 : Newspaper vs Sales
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Newspaper"],
    df["Sales"],
    color="orange"
)

plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("newspaper_vs_sales.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 4 : Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    color="red"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="black",
    linewidth=2
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)

plt.savefig("actual_vs_predicted_sales.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 5 : Correlation Heatmap
# ==========================================

plt.figure(figsize=(7,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 6 : Feature Importance (Coefficients)
# ==========================================

importance = pd.Series(
    model.coef_,
    index=X.columns
)

plt.figure(figsize=(8,6))

importance.plot(
    kind="bar",
    color=["blue", "green", "orange"]
)

plt.title("Feature Importance")
plt.xlabel("Advertising Medium")
plt.ylabel("Coefficient Value")
plt.grid(True)

plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n========================================")
print(" Project Completed Successfully ")
print("========================================")

print("\nGenerated PNG Files:")

print("1. tv_vs_sales.png")
print("2. radio_vs_sales.png")
print("3. newspaper_vs_sales.png")
print("4. actual_vs_predicted_sales.png")
print("5. correlation_heatmap.png")
print("6. feature_importance.png")