#!/usr/bin/env python3
"""
Step 2: Architecture 2 — Chemometrics (PLS-DA + PCA→LDA).

Reads the pre-split train/test CSVs, runs:
  1. PCA (n_components 2–10 by CV) → PLS-DA
  2. PCA → LDA (variant)
  3. PCA scatter plot (first 2 PCs, full dataset, coloured by material)

Outputs:
  models/arch2_plsda.pkl
  plots/pca_scatter.png
  plots/confusion_arch2_plsda.png
  plots/confusion_arch2_pcalda.png
  results/classification_reports.txt   (appended)

Usage:
  python analysis/nir_classification/02_train_arch2.py
  python analysis/nir_classification/02_train_arch2.py --savgol
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from nir_utils import PLSDAClassifier

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
    mat_train = df_train["Material_Type"].values
    mat_test  = df_test["Material_Type"].values

    if use_savgol:
        print("  Applying Savitzky-Golay 1st derivative...")
        X_train = apply_savgol(X_train)
        X_test  = apply_savgol(X_test)

    return X_train, X_test, y_train, y_test, mat_train, mat_test, le


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


def pca_scatter(X_train, X_test, mat_train, mat_test, class_names):
    """PCA scatter on the full dataset (train + test combined), coloured by material."""
    print("\n--- PCA scatter plot ---")
    X_all   = np.vstack([X_train, X_test])
    mat_all = np.concatenate([mat_train, mat_test])

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X_all)
    var = pca.explained_variance_ratio_

    palette = sns.color_palette("tab10", len(class_names))
    color_map = {name: palette[i] for i, name in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in class_names:
        mask = mat_all == name
        ax.scatter(Z[mask, 0], Z[mask, 1], label=name,
                   color=color_map[name], alpha=0.6, s=30)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("PCA Scatter — NIR Spectra (SNV, all samples)")
    ax.legend(title="Material")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "pca_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'pca_scatter.png'}")
    print(f"  PC1 explains {var[0]*100:.1f}%, PC2 explains {var[1]*100:.1f}% of variance")


def choose_pca_components(X_train, y_train):
    """Select best n_components (2–10) for PCA by 5-fold CV accuracy with LDA."""
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for n in range(2, 11):
        pipe = Pipeline([
            ("pca", PCA(n_components=n, random_state=42)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        score = cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1).mean()
        scores[n] = score
        print(f"    n_components={n}: CV accuracy={score:.4f}")

    best_n = max(scores, key=scores.get)
    print(f"  Best n_components: {best_n} (CV acc={scores[best_n]:.4f})")
    return best_n


def train_plsda(X_train, y_train, X_test, y_test, class_names, best_pca_n):
    print("\n--- PCA → PLS-DA ---")
    n_classes = len(class_names)
    n_pls = min(best_pca_n, n_classes - 1, 10)

    clf = PLSDAClassifier(n_pca=best_pca_n, n_pls=n_pls, n_classes=n_classes)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  PCA n_components={best_pca_n}, PLS n_components={n_pls}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Train time: {train_time:.2f}s  |  Inference: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_confusion(cm, class_names, "Arch 2 — PLS-DA", PLOT_DIR / "confusion_arch2_plsda.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "arch2_plsda.pkl")
    print(f"  Saved: {MODEL_DIR / 'arch2_plsda.pkl'}")

    text = (
        "=" * 60 + "\n"
        "ARCH 2 — PCA → PLS-DA\n"
        "=" * 60 + "\n"
        f"PCA n_components: {best_pca_n}\n"
        f"PLS n_components: {n_pls}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.2f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, train_time, infer_time


def train_pca_lda(X_train, y_train, X_test, y_test, class_names, best_pca_n):
    print("\n--- PCA → LDA ---")

    pipe = Pipeline([
        ("pca", PCA(n_components=best_pca_n, random_state=42)),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  PCA n_components={best_pca_n}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Train time: {train_time:.2f}s  |  Inference: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_confusion(cm, class_names, "Arch 2 — PCA → LDA", PLOT_DIR / "confusion_arch2_pcalda.png")

    joblib.dump(pipe, MODEL_DIR / "arch2_pcalda.pkl")
    print(f"  Saved: {MODEL_DIR / 'arch2_pcalda.pkl'}")

    text = (
        "=" * 60 + "\n"
        "ARCH 2 — PCA → LDA\n"
        "=" * 60 + "\n"
        f"PCA n_components: {best_pca_n}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.2f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, train_time, infer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savgol", action="store_true",
                        help="Apply Savitzky-Golay 1st derivative on top of SNV")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NIR CLASSIFICATION — Step 2: Architecture 2 (Chemometrics)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, mat_train, mat_test, le = load_data(args.savgol)
    class_names = list(le.classes_)
    print(f"\nClasses: {class_names}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pca_scatter(X_train, X_test, mat_train, mat_test, class_names)

    print("\nSelecting best PCA n_components (2–10) via cross-validation...")
    best_pca_n = choose_pca_components(X_train, y_train)

    train_plsda(X_train, y_train, X_test, y_test, class_names, best_pca_n)
    train_pca_lda(X_train, y_train, X_test, y_test, class_names, best_pca_n)

    print("\nArch 2 complete. Run 03_train_arch3.py next.\n")


if __name__ == "__main__":
    main()
