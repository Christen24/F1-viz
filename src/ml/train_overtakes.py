"""
F1 Visualization System — Overtake Detection Model Training

Trains an XGBoost binary classifier on overtake features.
Targets: Precision >= 0.8, Recall >= 0.7

Usage:
    python src/ml/train_overtakes.py --session 2024_Monza_R [--seed 42]
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(labeled_dir: Path, session: str):
    """Load features and labels, merge into training DataFrame."""
    features_path = labeled_dir / f"{session}_features.csv"
    labels_path = labeled_dir / f"{session}_heuristic_overtakes.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    features = pd.read_csv(features_path)

    # Create label column: default 0 (no overtake), 1 where we have labels
    features["label"] = 0

    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        # Match labels to features by time proximity and driver pair
        for _, label_row in labels.iterrows():
            t = label_row["t"]
            overtaker = label_row["overtaker"]
            overtaken = label_row["overtaken"]

            mask = (
                (abs(features["t"] - t) < 2.0) &
                (
                    ((features["driver_a"] == overtaker) & (features["driver_b"] == overtaken)) |
                    ((features["driver_a"] == overtaken) & (features["driver_b"] == overtaker))
                )
            )
            features.loc[mask, "label"] = 1

    logger.info("Data: %d samples, %d positive, %d negative",
                len(features), features["label"].sum(), (features["label"] == 0).sum())

    return features


def train_model(features: pd.DataFrame, seed: int):
    """Train XGBoost classifier and evaluate."""
    # Feature columns (exclude non-feature cols)
    feature_cols = [
        "relative_distance", "speed_diff", "gap_change_2s",
        "throttle_a", "brake_a", "speed_a", "drs_a",
    ]

    X = features[feature_cols].values
    y = features["label"].values

    # Handle class imbalance
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if pos_count >= 2 else None,
    )

    logger.info("Train: %d, Test: %d, Positive ratio: %.3f",
                len(X_train), len(X_test), y.mean())

    # Train
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("Results: P=%.3f, R=%.3f, F1=%.3f", precision, recall, f1)
    logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    logger.info("Feature importances: %s",
                {k: round(v, 3) for k, v in sorted(importances.items(), key=lambda x: -x[1])})

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "positive_ratio": round(float(y.mean()), 4),
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train overtake detection model")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    labeled_dir = data_dir / "labeled"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = data_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features = load_data(labeled_dir, args.session)

    if len(features) == 0:
        logger.error("No feature data found")
        sys.exit(1)

    # Train
    model, metrics = train_model(features, args.seed)

    # Save model
    model_path = models_dir / "overtake_detector_v1.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # Save experiment
    experiment = {
        "model": "overtake_detector_v1",
        "session": args.session,
        "seed": args.seed,
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "target_precision": 0.8,
        "target_recall": 0.7,
        "meets_targets": metrics["precision"] >= 0.8 and metrics["recall"] >= 0.7,
    }

    exp_path = experiments_dir / f"overtake_v1_{args.session}.json"
    with open(exp_path, "w") as f:
        json.dump(experiment, f, indent=2)

    # Update manifest
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": "1.0.0", "datasets": {}, "models": {}}

    manifest["models"]["overtake_detector_v1"] = {
        "session": args.session,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("✓ Training complete. Meets targets: %s", experiment["meets_targets"])


if __name__ == "__main__":
    main()
