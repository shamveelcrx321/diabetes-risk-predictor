# train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# 1. Load Dataset
# -----------------------------

data = pd.read_csv("data/diabetes.csv")


# -----------------------------
# 2. Data Cleaning
# Replace NaN values with median
# -----------------------------

columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns_to_clean:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)


# -----------------------------
# 3. Feature & Target Split
# -----------------------------

X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# -----------------------------
# 4. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # important for imbalanced dataset
)


# -----------------------------
# 5. Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 6. Tuned Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight={0: 1, 1: 2},
    random_state=42
)

model.fit(X_train_scaled, y_train)


# -----------------------------
# 7. Evaluation
# -----------------------------

y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 8. Save Model & Scaler
# -----------------------------

joblib.dump(model, "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler_dia.pkl")

print("\nModel and scaler saved successfully.")
