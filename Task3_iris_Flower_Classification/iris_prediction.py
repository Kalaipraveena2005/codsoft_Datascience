# ============================================================
# IRIS FLOWER CLASSIFICATION
# CODSOFT DATA SCIENCE INTERNSHIP - TASK 3
# ============================================================

# ==========================
# Import Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("IRIS.csv")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

# ============================================================
# Encode Target Column
# ============================================================

le = LabelEncoder()

df["species"] = le.fit_transform(df["species"])

print("\nEncoded Classes:")
print(le.classes_)

# ============================================================
# Features and Target
# ============================================================

X = df.drop("species", axis=1)
y = df["species"]

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

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted Classes:")
print(y_pred)

# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))

# ============================================================
# Graph 1 : Confusion Matrix
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
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
# Graph 3 : Model Accuracy
# ============================================================

plt.figure(figsize=(5,5))

plt.bar(
    ["Accuracy"],
    [accuracy],
    color="green"
)

plt.ylim(0,1.1)

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.savefig(
    "model_accuracy.png",
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

1. confusion_matrix.png
2. feature_importance.png
3. model_accuracy.png

All PNG files are saved in the project folder.
""")