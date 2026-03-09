"""
Kenya SME Credit Scoring - One-time setup script
Run this after cloning: python setup.py
It will generate data (if needed) and train + save the model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pathlib import Path
import joblib
import sys

print("=" * 50)
print("  CreditIQ Kenya — Setup")
print("=" * 50)

# ── Step 1: Data ──
data_path = Path("data/kenya_sme_dataset.csv")
if not data_path.exists():
    print("\n[1/2] Generating dataset...")
    from generate_data import generate
    df = generate()
else:
    df = pd.read_csv(data_path)
    print(f"\n[1/2] Dataset found: {df.shape[0]} rows")

# ── Step 2: Train ──
print("\n[2/2] Training model...")

X = df.drop(columns=["company_id", "credit_default"])
y = df["credit_default"]

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"    Accuracy : {accuracy:.2%}")
print(f"    numpy    : {np.__version__}")
print(f"    sklearn  : {__import__('sklearn').__version__}")
print()
print(classification_report(y_test, model.predict(X_test), target_names=["No Default", "Default"]))

# ── Step 3: Save ──
Path("models").mkdir(exist_ok=True)
joblib.dump(model,          "models/kenya_sme_credit_model.pkl")
joblib.dump(list(X.columns),"models/feature_columns.pkl")

print("✅ Model saved to models/")
print("\n  Run the app with:")
print("      streamlit run app.py\n")
