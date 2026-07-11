# ============================================================
# CREDIT CARD FRAUD DETECTION
# CODSOFT DATA SCIENCE INTERNSHIP - TASK 5
# ============================================================

# ==========================
# Import Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Load Dataset
# ============================================================
df = pd.read_csv("creditcard.csv")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Fraud Distribution ==========")
print(df["Class"].value_counts())

# ============================================================
# Graph 1 : Fraud Distribution
# ============================================================
plt.figure(figsize=(6,5))

sns.countplot(
    x="Class",
    data=df,
    palette="Set2"
)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig(
    "fraud_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Graph 2 : Transaction Amount Distribution
# ============================================================
plt.figure(figsize=(8,5))

plt.hist(
    df["Amount"],
    bins=50,
    color="skyblue",
    edgecolor="black"
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.savefig(
    "transaction_amount_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Graph 3 : Correlation Heatmap
# ============================================================
plt.figure(figsize=(12,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Features and Target
# ============================================================
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# Split Dataset
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ============================================================
# Create Model
# ============================================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ============================================================
# Train Model
# ============================================================
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# ============================================================
# Prediction
# ============================================================
y_pred = model.predict(X_test)

print("\nFirst 20 Predictions")
print(y_pred[:20])

# ============================================================
# Accuracy
# ============================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy :", accuracy)

# ============================================================
# Confusion Matrix
# ============================================================
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred
))

# ============================================================
# Graph 4 : Confusion Matrix
# ============================================================
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Fraud"],
    yticklabels=["Normal","Fraud"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Graph 5 : Feature Importance
# ============================================================
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(10,8))

importance.sort_values().plot(
    kind="barh",
    color="teal"
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.grid(True)

plt.savefig(
    "feature_importance.png",
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

1. fraud_distribution.png
2. transaction_amount_distribution.png
3. correlation_heatmap.png
4. confusion_matrix.png
5. feature_importance.png

All PNG files are saved in the project folder.
""")