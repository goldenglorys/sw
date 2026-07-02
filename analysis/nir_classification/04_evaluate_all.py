#!/usr/bin/env python3
"""
Step 4: Unified evaluation — load all saved models and compare on the test set.

Run this AFTER all three training scripts have completed. It:
  - Loads each saved model
  - Re-runs inference on the test set
  - Prints per-class metrics for every model
  - Writes results/comparison_table.csv  (accuracy, macro-F1, infer time)
  - Highlights the best model per metric

Outputs:
  results/comparison_table.csv
  results/classification_reports.txt   (appended with final summary)

Usage:
  python analysis/nir_classification/04_evaluate_all.py
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from scipy.signal import savgol_filter
from nir_utils import PLSDAClassifier  # noqa: F401 — needed so pickle can find the class

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
N_FEATURES = len(SNV_COLS)


def load_test_data():
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    le      = joblib.load(PREP_DIR / "label_encoder.pkl")
    X_test  = df_test[SNV_COLS].values.astype(np.float32)
    y_test  = df_test["label_idx"].values
    return X_test, y_test, le


def measure_inference(predict_fn, X, n_reps=5):
    """Average inference time per sample over n_reps runs (ms)."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return (np.mean(times) / len(X)) * 1000


def evaluate_sklearn_model(name, model_path, X_test, y_test, class_names):
    if not model_path.exists():
        print(f"  SKIP {name}: model file not found at {model_path}")
        return None

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    infer_ms = measure_inference(model.predict, X_test)

    acc    = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  {name}")
    print(f"  Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Infer: {infer_ms:.4f} ms/sample")
    print(f"\n{report}")

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "infer_ms_per_sample": infer_ms,
        "y_pred": y_pred,
        "cm": cm,
        "report": report,
    }


def evaluate_keras_model(name, model_path, X_test, y_test, class_names, reshape=False):
    if not model_path.exists():
        print(f"  SKIP {name}: model file not found at {model_path}")
        return None

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    X_in = X_test.reshape(-1, N_FEATURES, 1) if reshape else X_test

    def pred_fn(X):
        return np.argmax(model.predict(X, verbose=0), axis=1)

    y_pred    = pred_fn(X_in)
    infer_ms  = measure_inference(pred_fn, X_in)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report   = classification_report(y_test, y_pred, target_names=class_names)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n  {name}")
    print(f"  Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Infer: {infer_ms:.4f} ms/sample")
    print(f"\n{report}")

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "infer_ms_per_sample": infer_ms,
        "y_pred": y_pred,
        "cm": cm,
        "report": report,
    }


def plot_confusion(result, class_names):
    safe_name = result["model"].replace(" ", "_").replace("→", "to").replace("/", "_")
    save_path = PLOT_DIR / f"eval_{safe_name}_confusion.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        result["cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion — {result['model']}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_comparison_table(results: list):
    rows = [{
        "Model":               r["model"],
        "Accuracy":            f"{r['accuracy']:.4f}",
        "Macro F1":            f"{r['macro_f1']:.4f}",
        "Infer (ms/sample)":   f"{r['infer_ms_per_sample']:.4f}",
    } for r in results]

    df = pd.DataFrame(rows)

    # Find best per metric
    accs     = [r["accuracy"]           for r in results]
    f1s      = [r["macro_f1"]           for r in results]
    infers   = [r["infer_ms_per_sample"] for r in results]

    best_acc_i   = int(np.argmax(accs))
    best_f1_i    = int(np.argmax(f1s))
    best_infer_i = int(np.argmin(infers))

    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    print(f"  Best accuracy  : {results[best_acc_i]['model']} ({accs[best_acc_i]:.4f})")
    print(f"  Best macro-F1  : {results[best_f1_i]['model']} ({f1s[best_f1_i]:.4f})")
    print(f"  Fastest infer  : {results[best_infer_i]['model']} ({infers[best_infer_i]:.4f} ms/sample)")
    print("=" * 70)

    return df, best_acc_i, best_f1_i, best_infer_i


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NIR CLASSIFICATION — Step 4: Evaluate All Models")
    print("=" * 60)

    X_test, y_test, le = load_test_data()
    class_names = list(le.classes_)
    print(f"\nTest set: {len(X_test)} samples")
    print(f"Classes: {class_names}")

    results = []

    print("\n--- Architecture 1: Classical ML ---")
    for name, path in [
        ("Arch1 RF",  MODEL_DIR / "arch1_rf.pkl"),
        ("Arch1 SVM", MODEL_DIR / "arch1_svm.pkl"),
    ]:
        r = evaluate_sklearn_model(name, path, X_test.astype(np.float64), y_test, class_names)
        if r:
            results.append(r)

    print("\n--- Architecture 2: Chemometrics ---")
    for name, path in [
        ("Arch2 PLS-DA",  MODEL_DIR / "arch2_plsda.pkl"),
        ("Arch2 PCA-LDA", MODEL_DIR / "arch2_pcalda.pkl"),
    ]:
        r = evaluate_sklearn_model(name, path, X_test.astype(np.float64), y_test, class_names)
        if r:
            results.append(r)

    print("\n--- Architecture 3: Neural Networks ---")
    r = evaluate_keras_model(
        "Arch3 MLP", MODEL_DIR / "arch3_mlp.keras",
        X_test, y_test, class_names, reshape=False
    )
    if r:
        results.append(r)

    r = evaluate_keras_model(
        "Arch3 CNN", MODEL_DIR / "arch3_cnn.keras",
        X_test, y_test, class_names, reshape=True
    )
    if r:
        results.append(r)

    if not results:
        print("\nNo models found. Run training scripts first.")
        return

    df_table, best_acc_i, best_f1_i, best_infer_i = print_comparison_table(results)

    # Save comparison table
    csv_path = RESULT_DIR / "comparison_table.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Append summary to classification_reports.txt
    summary = (
        "=" * 60 + "\n"
        "FINAL COMPARISON TABLE\n"
        "=" * 60 + "\n"
        + df_table.to_string(index=False) + "\n\n"
        f"Best accuracy : {results[best_acc_i]['model']}\n"
        f"Best macro-F1 : {results[best_f1_i]['model']}\n"
        f"Fastest infer : {results[best_infer_i]['model']}\n\n"
    )
    with open(RESULT_DIR / "classification_reports.txt", "a") as f:
        f.write(summary)
    print(f"Summary appended to: {RESULT_DIR / 'classification_reports.txt'}")

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
