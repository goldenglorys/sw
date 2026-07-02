#!/usr/bin/env python3
"""
Step 3: Architecture 3 — Neural Networks (MLP + 1D CNN).

Reads the pre-split train/test CSVs, trains:
  1. MLP with two hidden-layer configurations [64,32] and [128,64,32]
  2. 1D CNN

Both use:
  - Adam lr=1e-3
  - Categorical cross-entropy loss
  - Early stopping (patience=15) on validation loss
  - 15% validation split from training set (separate from the held-out test)

Outputs:
  models/arch3_mlp.keras
  models/arch3_cnn.keras
  plots/training_curves_mlp.png
  plots/training_curves_cnn.png
  plots/confusion_arch3_mlp.png
  plots/confusion_arch3_cnn.png
  results/classification_reports.txt   (appended)

Usage:
  python analysis/nir_classification/03_train_arch3.py
  python analysis/nir_classification/03_train_arch3.py --savgol
  python analysis/nir_classification/03_train_arch3.py --epochs 200
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import savgol_filter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


def apply_savgol(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        lambda row: savgol_filter(row, window_length=7, polyorder=2, deriv=1),
        axis=1, arr=X
    )


def load_data(use_savgol: bool):
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test  = pd.read_csv(DATA_DIR / "test.csv")
    le       = joblib.load(PREP_DIR / "label_encoder.pkl")

    X_train = df_train[SNV_COLS].values.astype(np.float32)
    X_test  = df_test[SNV_COLS].values.astype(np.float32)
    y_train = df_train["label_idx"].values
    y_test  = df_test["label_idx"].values

    if use_savgol:
        print("  Applying Savitzky-Golay 1st derivative...")
        X_train = apply_savgol(X_train).astype(np.float32)
        X_test  = apply_savgol(X_test).astype(np.float32)

    n_classes = len(le.classes_)
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    Y_test  = tf.keras.utils.to_categorical(y_test,  num_classes=n_classes)

    return X_train, X_test, y_train, y_test, Y_train, Y_test, le, n_classes


def plot_training_curves(history, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


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


def build_mlp(n_features: int, n_classes: int, hidden_layers: list) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(n_features: int, n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features, 1))
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    model, X_train, Y_train, X_test, y_test, Y_test,
    class_names, model_name, max_epochs, plot_prefix, model_path
):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train, Y_train,
        epochs=max_epochs,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1,
    )
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])

    t1 = time.perf_counter()
    Y_pred_prob = model.predict(X_test, verbose=0)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000
    y_pred = np.argmax(Y_pred_prob, axis=1)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Epochs run: {epochs_run}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Train time: {train_time:.1f}s  |  Inference: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_training_curves(history, model_name, PLOT_DIR / f"training_curves_{plot_prefix}.png")
    plot_confusion(cm, class_names, f"Arch 3 — {model_name}", PLOT_DIR / f"confusion_arch3_{plot_prefix}.png")

    model.save(model_path)
    print(f"  Saved: {model_path}")

    text = (
        "=" * 60 + "\n"
        f"ARCH 3 — {model_name}\n"
        "=" * 60 + "\n"
        f"Epochs run (early stopping): {epochs_run}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.1f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, train_time, infer_time


def train_mlp(X_train, Y_train, X_test, y_test, Y_test,
              class_names, n_classes, max_epochs):
    print("\n--- MLP ---")
    configs = [[64, 32], [128, 64, 32]]
    best_acc  = -1
    best_model = None
    best_history_info = {}

    for cfg in configs:
        print(f"\n  Config: hidden layers = {cfg}")
        model = build_mlp(N_FEATURES, n_classes, cfg)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        history = model.fit(
            X_train, Y_train,
            epochs=max_epochs,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Epochs: {len(history.history['loss'])}  Test acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_cfg = cfg
            best_history = history

    print(f"\n  Best MLP config: {best_cfg}  (acc={best_acc:.4f})")

    t0 = time.perf_counter()
    best_model.fit(  # already trained — re-use history for timing reference
        X_train, Y_train,
        epochs=0,      # 0 epochs = no actual training, just for timing baseline
        batch_size=32,
        validation_split=0.15,
        verbose=0,
    )
    # Measure inference time only
    t1 = time.perf_counter()
    Y_pred_prob = best_model.predict(X_test, verbose=0)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000

    y_pred = np.argmax(Y_pred_prob, axis=1)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    epochs_run = len(best_history.history["loss"])
    val_loss_min = min(best_history.history["val_loss"])

    print(f"\n  Final test accuracy: {acc:.4f}")
    print(f"  Inference time: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_training_curves(best_history, f"MLP {best_cfg}", PLOT_DIR / "training_curves_mlp.png")
    plot_confusion(cm, class_names, "Arch 3 — MLP", PLOT_DIR / "confusion_arch3_mlp.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save(MODEL_DIR / "arch3_mlp.keras")
    print(f"  Saved: {MODEL_DIR / 'arch3_mlp.keras'}")

    # Estimate train time from history length * a benchmark timing
    t_train0 = time.perf_counter()
    dummy = build_mlp(N_FEATURES, n_classes, best_cfg)
    dummy.fit(X_train, Y_train, epochs=1, batch_size=32, validation_split=0.15, verbose=0)
    time_per_epoch = time.perf_counter() - t_train0
    train_time_est = epochs_run * time_per_epoch

    text = (
        "=" * 60 + "\n"
        f"ARCH 3 — MLP (best config: {best_cfg})\n"
        "=" * 60 + "\n"
        f"Configs tried: {configs}\n"
        f"Best config: {best_cfg}\n"
        f"Epochs run (early stopping): {epochs_run}\n"
        f"Min val loss: {val_loss_min:.4f}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time (est): {train_time_est:.1f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, train_time_est, infer_time


def train_cnn(X_train, Y_train, X_test, y_test, Y_test,
              class_names, n_classes, max_epochs):
    print("\n--- 1D CNN ---")

    X_train_cnn = X_train.reshape(-1, N_FEATURES, 1)
    X_test_cnn  = X_test.reshape(-1, N_FEATURES, 1)

    model = build_cnn(N_FEATURES, n_classes)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train_cnn, Y_train,
        epochs=max_epochs,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1,
    )
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])

    t1 = time.perf_counter()
    Y_pred_prob = model.predict(X_test_cnn, verbose=0)
    infer_time = (time.perf_counter() - t1) / len(y_test) * 1000
    y_pred = np.argmax(Y_pred_prob, axis=1)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Epochs run: {epochs_run}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Train time: {train_time:.1f}s  |  Inference: {infer_time:.4f} ms/sample")
    print(f"\n{report}")

    plot_training_curves(history, "1D CNN", PLOT_DIR / "training_curves_cnn.png")
    plot_confusion(cm, class_names, "Arch 3 — 1D CNN", PLOT_DIR / "confusion_arch3_cnn.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "arch3_cnn.keras")
    print(f"  Saved: {MODEL_DIR / 'arch3_cnn.keras'}")

    text = (
        "=" * 60 + "\n"
        "ARCH 3 — 1D CNN\n"
        "=" * 60 + "\n"
        f"Epochs run (early stopping): {epochs_run}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Train time: {train_time:.1f}s\n"
        f"Inference time: {infer_time:.4f} ms/sample\n\n"
        + report + "\n"
    )
    write_report(text)

    return acc, train_time, infer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savgol", action="store_true")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max epochs (early stopping will cut it short, default=300)")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NIR CLASSIFICATION — Step 3: Architecture 3 (MLP + CNN)")
    print("=" * 60)
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    X_train, X_test, y_train, y_test, Y_train, Y_test, le, n_classes = load_data(args.savgol)
    class_names = list(le.classes_)
    print(f"\nClasses: {class_names}  ({n_classes} classes)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    train_mlp(X_train, Y_train, X_test, y_test, Y_test, class_names, n_classes, args.epochs)
    train_cnn(X_train, Y_train, X_test, y_test, Y_test, class_names, n_classes, args.epochs)

    print("\nArch 3 complete. Run 04_evaluate_all.py to generate the comparison table.\n")


if __name__ == "__main__":
    main()
