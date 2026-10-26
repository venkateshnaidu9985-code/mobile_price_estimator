"""Train a Linear Regression model that estimates a smartphone's price and save it to model.pkl.

Usage:
    python train_model.py                   # uses a synthetic sample dataset
    python train_model.py path/to/data.csv  # CSV with the feature columns below plus `price`
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"

FEATURES = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
            "screen_inch", "refresh_rate", "has_5g", "brand_tier"]
TARGET = "price"


def build_sample_dataset(n=1500, seed=42):
    """Synthetic phone specs with a realistic-looking price (USD) plus noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ram_gb": rng.choice([3, 4, 6, 8, 12, 16], n),
        "storage_gb": rng.choice([32, 64, 128, 256, 512, 1024], n),
        "battery_mah": rng.integers(3000, 6001, n),
        "camera_mp": rng.choice([8, 12, 13, 16, 32, 48, 50, 64, 108, 200], n),
        "screen_inch": np.round(rng.uniform(5.0, 7.2, n), 1),
        "refresh_rate": rng.choice([60, 90, 120, 144], n),
        "has_5g": rng.integers(0, 2, n),
        "brand_tier": rng.choice([1, 2, 3], n),  # 1 = budget, 2 = mid, 3 = premium
    })
    price = (
        40
        + 14 * df["ram_gb"]
        + 0.35 * df["storage_gb"]
        + 0.02 * df["battery_mah"]
        + 1.1 * df["camera_mp"]
        + 25 * df["screen_inch"]
        + 1.2 * df["refresh_rate"]
        + 55 * df["has_5g"]
        + 90 * df["brand_tier"]
        + rng.normal(0, 25, n)
    )
    df[TARGET] = price.clip(lower=60).round(0)
    return df


def load_dataset(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        return df[FEATURES + [TARGET]].dropna()
    return build_sample_dataset()


def build_pipeline():
    return Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} phones")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    model = build_pipeline().fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R^2: {r2_score(y_test, preds):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.1f}")

    coefs = dict(zip(FEATURES, model.named_steps["lr"].coef_.round(2)))
    print("Scaled coefficients:", coefs)

    model.fit(df[FEATURES], df[TARGET])  # refit on all data
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
