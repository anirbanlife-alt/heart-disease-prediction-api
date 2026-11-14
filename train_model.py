from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
CSV_PATH = DATA_DIR / "heart.csv"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"

# If the CSV is not in data/, this public copy is downloaded automatically.
# You can instead put the CSV downloaded from the Kaggle assignment link here.
DATA_URL = "https://raw.githubusercontent.com/DATAHUB-KIT/datasets/main/heart.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    if not CSV_PATH.exists():
        print("heart.csv was not found. Downloading a compatible copy...")
        urlretrieve(DATA_URL, CSV_PATH)

    data = pd.read_csv(CSV_PATH)
    data.columns = data.columns.str.strip().str.lower()

    # The assignment dataset normally uses 'target'. Some versions use 'num'.
    target_column = "target" if "target" in data.columns else "num"
    if target_column not in data.columns:
        raise ValueError("Could not find the target column. Expected 'target' or 'num'.")

    missing = [name for name in FEATURES if name not in data.columns]
    if missing:
        raise ValueError(f"These required columns are missing from heart.csv: {missing}")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[target_column], errors="coerce")

    # In the original Cleveland data, values 1-4 mean disease is present.
    # In the Kaggle version, target is already 0 or 1.
    clean = pd.concat([X, y.rename("label")], axis=1).dropna(subset=["label"])
    X = clean[FEATURES]
    y = (clean["label"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Missing values are filled automatically; the model is kept in one pipeline.
    model = Pipeline([
        ("fill_missing_values", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    joblib.dump(model, MODEL_PATH)

    print("\nTraining is complete.")
    print(f"Rows used: {len(clean)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nA quick look at the test results:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
