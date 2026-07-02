#!/usr/bin/env python3
"""
Step 1: Architecture 1 — Classical ML (Random Forest + SVM).

Reads the pre-split train/test CSVs produced by 00_split_and_preprocess.py,
runs GridSearchCV (5-fold stratified CV) on the training set for both models,
evaluates on the held-out test set, and saves:

  models/arch1_rf.pkl
  models/arch1_svm.pkl
  plots/confusion_arch1_rf.png
  plots/confusion_arch1_svm.png
  results/classification_reports.txt   (appended)

Usage:
  python analysis/nir_classification/01_train_arch1.py
  python analysis/nir_classification/01_train_arch1.py --savgol   # with SG derivative
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from scipy.signal import savgol_filter

BASE = Path(__file__).resolve().parent
DATA_DIR   = BASE / "data"
PREP_DIR   = BASE / "preprocessing"
MODEL_DIR  = BASE / "models"
PLOT_DIR   = BASE / "plots"
RESULT_DIR = BASE / "results"

SNV_COLS = [
    "SNV_NIR_410nm", "SNV_NIR_435nm", "SNV_NIR_460nm", "SNV_NIR_485nm",
    "SNV_NIR_510nm", "SNV_NIR_535nm", "SNV_NIR_560nm", "SNV_NIR_585nm",
    "SNV_NIR_610nm", "SNV_NIR_645nm", "SNV_NIR_680nm", "SNV_NIR_705nm",
    "SNV_NIR_730nm", "SNV_NIR_760nm", "SNV_NIR_810nm", "SNV_NIR_860nm",
    "SNV_NIR_900nm", "SNV_NIR_940nm",
]


def apply_savgol(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay 1st derivative (window=7, poly=2) on SNV spectra."""
    return np.apply_along_axis(
        lambda row: savgol_filter(row, window_length=7, polyorder=2, deriv=1),
        axis=1, arr=X
    )


def load_data(use_savgol: bool):
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test  = pd.read_csv(DATA_DIR / "test.csv")
    le       = joblib.load(PREP_DIR / "label_encoder.pkl")

    X_train = df_train[SNV_COLS].values.astype(np.float64)
    X_test  = df_test[SNV_COLS].values.astype(np.float64)
    y_train = df_train["label_idx"].values
    y_test  = df_test["label_idx"].values

    if use_savgol:
        print("  Applying Savitzky-Golay 1st derivative...")
        X_train = apply_savgol(X_train)
        X_test  = apply_savgol(X_test)

    return X_train, X_test, y_train, y_test, le


def plot_confusion(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def write_report(text: str):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "classification_reports.txt", "a") as f:
        f.write(text + "\n")


def train_random_forest(X_train, y_train, X_test, y_test, class_names):
    print("\n--- Random Forest ---")
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth":    [None, 5, 10, 20],
        "max_features": ["sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}")
    print(f"  Training time: {train_time:.1f}s")

    best = grid.best_estimator_

    t1 = time.perf_counter()
    y_pred = best.predict(X_test)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000  # ms/sample

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Test accuracy: {acc:.4f}")
    print(f"  Inference time: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_confusion(cm, class_names, "Arch 1 — Random Forest", PLOT_DIR / "confusion_arch1_rf.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "arch1_rf.pkl")
    print(f"  Saved: {MODEL_DIR / 'arch1_rf.pkl'}")

    text = (
        "=" * 60 + "\n"
        "ARCH 1 — Random Forest\n"
        "=" * 60 + "\n"
        f"Best params: {grid.best_params_}\n"
        f"CV accuracy: {grid.best_score_:.4f}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.1f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, report, train_time, infer_time


def train_svm(X_train, y_train, X_test, y_test, class_names):
    print("\n--- SVM (RBF kernel) ---")
    param_grid = {
        "C":     [0.1, 1, 10, 100, 1000],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}")
    print(f"  Training time: {train_time:.1f}s")

    best = grid.best_estimator_

    t1 = time.perf_counter()
    y_pred = best.predict(X_test)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Test accuracy: {acc:.4f}")
    print(f"  Inference time: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_confusion(cm, class_names, "Arch 1 — SVM (RBF)", PLOT_DIR / "confusion_arch1_svm.png")

    joblib.dump(best, MODEL_DIR / "arch1_svm.pkl")
    print(f"  Saved: {MODEL_DIR / 'arch1_svm.pkl'}")

    text = (
        "=" * 60 + "\n"
        "ARCH 1 — SVM (RBF kernel)\n"
        "=" * 60 + "\n"
        f"Best params: {grid.best_params_}\n"
        f"CV accuracy: {grid.best_score_:.4f}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.1f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, report, train_time, infer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savgol", action="store_true",
                        help="Apply Savitzky-Golay 1st derivative on top of SNV")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NIR CLASSIFICATION — Step 1: Architecture 1 (RF + SVM)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le = load_data(args.savgol)
    class_names = list(le.classes_)
    print(f"\nClasses: {class_names}")
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    train_random_forest(X_train, y_train, X_test, y_test, class_names)
    train_svm(X_train, y_train, X_test, y_test, class_names)

    print("\nArch 1 complete. Run 02_train_arch2.py next.\n")


if __name__ == "__main__":
    main()
