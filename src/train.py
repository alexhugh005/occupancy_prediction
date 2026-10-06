"""
Training pipeline for Occupancy Prediction model.
Implements: data load -> feature engineering (OHE from datetime) ->
Random Search HPO with XGBoost -> evaluation (ROC-AUC) -> model persistence.
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib

# Training commonly runs in a container or CI environment without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

from feature_engineering import FeatureEngineer, parse_datetime


def load_data(path: str) -> pd.DataFrame:
    """Load CSV, drop artificial index if present, parse types."""
    df = pd.read_csv(path)
    # Original files have a leading index-like first column sometimes named Unnamed
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    # Ensure Occupancy is int
    df["Occupancy"] = df["Occupancy"].astype(int)
    return df


def prepare_xy(df: pd.DataFrame, fe: FeatureEngineer, fit: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering and return X, y."""
    if fit:
        X = fe.fit_transform(df)
    else:
        X = fe.transform(df)
    y = df["Occupancy"].reset_index(drop=True)
    return X, y


def get_param_distributions() -> Dict[str, Any]:
    """Hyperparameter search space for Random Search."""
    return {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }


def save_roc_auc_plot(
    curves: Dict[str, Tuple[pd.Series, np.ndarray]], output_path: str
) -> None:
    """Save ROC curves for one or more evaluation data sets to ``output_path``."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="No-skill baseline")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic (ROC) Curve",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_pipeline(
    train_path: str,
    test_path: str | None = None,
    model_dir: str = "models",
    results_dir: str = "results",
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline:
    1. Load data
    2. Feature engineering (fit on train)
    3. Random Search HPO for XGBoost
    4. Refit best model
    5. Evaluate (ROC-AUC) on hold-out / provided test
    6. Persist model + feature engineer + metrics
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_dir, f"run_{timestamp}")
    results_run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_run_dir, exist_ok=True)

    print("=== 1. Data Ingestion ===")
    train_df = load_data(train_path)
    print(f"Training samples: {len(train_df)}")
    print(f"Occupancy distribution:\n{train_df['Occupancy'].value_counts()}")

    print("\n=== 2. Feature Engineering (fit) ===")
    fe = FeatureEngineer()
    X_train_full, y_train_full = prepare_xy(train_df, fe, fit=True)
    print(f"Feature matrix shape: {X_train_full.shape}")
    print(f"Features: {list(X_train_full.columns)[:8]}...")

    # Internal validation split for HPO evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_state, stratify=y_train_full
    )

    print("\n=== 3. Hyperparameter Optimization (Random Search) ===")
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
    )
    param_dist = get_param_distributions()
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    print("\n=== 4. Final Model Training ===")
    best_model = search.best_estimator_
    # Optionally refit on full train
    best_model.fit(X_train_full, y_train_full)

    print("\n=== 5. Evaluation ===")
    # Validation set metrics
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"Hold-out Validation ROC-AUC: {val_auc:.4f}")

    metrics = {
        "best_cv_roc_auc": float(search.best_score_),
        "val_roc_auc": float(val_auc),
        "best_params": search.best_params_,
        "n_features": X_train_full.shape[1],
        "train_samples": len(train_df),
        "timestamp": timestamp,
    }
    roc_curves = {"Validation": (y_val, y_val_proba)}

    # Optional external test set
    if test_path and os.path.exists(test_path):
        test_df = load_data(test_path)
        X_test, y_test = prepare_xy(test_df, fe, fit=False)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        print(f"External Test ROC-AUC: {test_auc:.4f}")
        metrics["test_roc_auc"] = float(test_auc)
        roc_curves["External test"] = (y_test, y_test_proba)
        y_pred = best_model.predict(X_test)
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    print("\n=== 6. Persist Artifacts ===")
    model_path = os.path.join(run_dir, "model.joblib")
    fe_path = os.path.join(run_dir, "feature_engineer.joblib")
    metrics_path = os.path.join(run_dir, "metrics.json")
    roc_plot_path = os.path.join(results_run_dir, "roc_auc_curve.png")
    # Also keep a "latest" symlink-like copy
    latest_model = os.path.join(model_dir, "model.joblib")
    latest_fe = os.path.join(model_dir, "feature_engineer.joblib")

    joblib.dump(best_model, model_path)
    fe.save(fe_path)
    save_roc_auc_plot(roc_curves, roc_plot_path)
    metrics["roc_auc_plot"] = roc_plot_path
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(best_model, latest_model)
    fe.save(latest_fe)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {run_dir}")
    print(f"Evaluation plots saved to {results_run_dir}")
    print(f"Also updated latest in {model_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Occupancy Prediction model")
    parser.add_argument(
        "--train-path",
        default="data/datatraining.txt",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--test-path",
        default="data/datatest.txt",
        help="Path to test CSV (optional)",
    )
    parser.add_argument("--model-dir", default="models", help="Directory to save model")
    parser.add_argument("--results-dir", default="results", help="Directory to save evaluation plots")
    parser.add_argument("--n-iter", type=int, default=8, help="Random search iterations")
    parser.add_argument("--cv", type=int, default=3, help="CV folds")
    args = parser.parse_args()

    # Resolve relative to project root if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_path = args.train_path if os.path.isabs(args.train_path) else os.path.join(project_root, args.train_path)
    test_path = args.test_path if os.path.isabs(args.test_path) else os.path.join(project_root, args.test_path)
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(project_root, args.model_dir)
    results_dir = args.results_dir if os.path.isabs(args.results_dir) else os.path.join(project_root, args.results_dir)

    train_pipeline(
        train_path=train_path,
        test_path=test_path,
        model_dir=model_dir,
        results_dir=results_dir,
        n_iter=args.n_iter,
        cv=args.cv,
    )
