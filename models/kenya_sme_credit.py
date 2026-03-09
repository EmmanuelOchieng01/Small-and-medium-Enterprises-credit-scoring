# Kenya SME Credit Scoring - Main Script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib

print("Kenya SME Credit Scoring Model")
print("=" * 40)

def load_data():
    """Load and return SME data"""
    try:
        df = pd.read_csv('data/kenya_sme_dataset.csv')
        print(f"✅ Data loaded: {df.shape[0]} SMEs, {df.shape[1]} features")
        print(f"   Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print("❌ Data file not found.")
        return None

def prepare_features(df):
    """
    Automatically detect feature columns and target column.
    Handles both numeric and categorical columns.
    """
    print("\n📊 Preparing features...")

    # Try common target column names
    target_candidates = [
        'credit_score', 'score', 'creditworthy', 'credit_risk',
        'default', 'label', 'target', 'risk', 'approved'
    ]
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        # Fall back to last column
        target_col = df.columns[-1]
        print(f"⚠️  No known target column found. Using last column: '{target_col}'")
    else:
        print(f"✅ Target column: '{target_col}'")

    # Feature columns = everything except target
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categorical features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"   Encoded categorical column: '{col}'")

    # Fill any missing values
    X = X.fillna(X.median(numeric_only=True))

    print(f"   Feature columns ({len(feature_cols)}): {feature_cols}")
    return X, y, feature_cols

def train_model(X, y):
    """Train a RandomForest model"""
    print("\n🤖 Training RandomForestClassifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained. Test accuracy: {accuracy:.2%}")
    return model

def save_model(model, feature_cols):
    """Save model and feature list"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "kenya_sme_credit_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to: {model_path}")

    # Also save feature column names so app.py knows what columns to send
    features_path = models_dir / "feature_columns.pkl"
    joblib.dump(feature_cols, features_path)
    print(f"✅ Feature columns saved to: {features_path}")

    return model_path

if __name__ == "__main__":
    print("🚀 Starting Kenya SME Credit Scoring System...")

    data = load_data()

    if data is not None:
        X, y, feature_cols = prepare_features(data)
        model = train_model(X, y)
        save_model(model, feature_cols)
        print("\n🎉 Done! You can now run: streamlit run app.py")
    else:
        print("❌ Could not load data. Exiting.")
