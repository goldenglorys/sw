#!/usr/bin/env python3
"""
Step 0: Data splitting and preprocessing.

Loads the raw NIR dataset, stratified 70/30 splits it, fits SNV normalisation
on the training set only, and saves all artefacts needed by training scripts.

Outputs:
  analysis/nir_classification/data/train.csv
  analysis/nir_classification/data/test.csv
  analysis/nir_classification/preprocessing/snv_params.pkl
  analysis/nir_classification/preprocessing/label_encoder.pkl
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nir_utils import SNVTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "nir" / "plastic_nir_dataset.csv"
OUT_BASE = Path(__file__).resolve().parent

DATA_DIR   = OUT_BASE / "data"
PREP_DIR   = OUT_BASE / "preprocessing"

NIR_COLS = [
    "NIR_410nm", "NIR_435nm", "NIR_460nm", "NIR_485nm",
    "NIR_510nm", "NIR_535nm", "NIR_560nm", "NIR_585nm",
    "NIR_610nm", "NIR_645nm", "NIR_680nm", "NIR_705nm",
    "NIR_730nm", "NIR_760nm", "NIR_810nm", "NIR_860nm",
    "NIR_900nm", "NIR_940nm",
]
TARGET_COL = "Material_Type"
RANDOM_SEED = 42


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, skiprows=1)
    for col in NIR_COLS + ["NIR_Temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = df[NIR_COLS].isnull().sum().sum()
    if missing > 0:
        print(f"  WARNING: {missing} NaN values in NIR columns — dropping those rows.")
        df = df.dropna(subset=NIR_COLS)
    return df


def main():
    print("=" * 60)
    print("  NIR CLASSIFICATION — Step 0: Split & Preprocess")
    print("=" * 60)

    df = load_raw()
    print(f"\nLoaded {len(df)} samples, {df[TARGET_COL].nunique()} material types")
    print("\nClass distribution:")
    for mat, cnt in df[TARGET_COL].value_counts().items():
        flag = "  << low sample count" if cnt < 100 else ""
        print(f"  {mat:<6}: {cnt:>4}{flag}")

    # Label encode
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df[TARGET_COL])
    label_map = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"\nLabel encoding: {label_map}")

    # Stratified 70/30 split
    X = df[NIR_COLS].values
    y = df["label_idx"].values
    idx = np.arange(len(df))

    idx_train, idx_test = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    print(f"\nSplit: {len(df_train)} train / {len(df_test)} test (seed={RANDOM_SEED})")
    print("\nPer-class split:")
    for mat in le.classes_:
        n_tr = (df_train[TARGET_COL] == mat).sum()
        n_te = (df_test[TARGET_COL]  == mat).sum()
        print(f"  {mat:<6}: train={n_tr:>4}, test={n_te:>4}")

    # Fit SNV on train features, apply to both
    X_train = df_train[NIR_COLS].values.astype(np.float64)
    X_test  = df_test[NIR_COLS].values.astype(np.float64)

    snv = SNVTransformer().fit(X_train)
    X_train_snv = snv.transform(X_train)
    X_test_snv  = snv.transform(X_test)

    # Write SNV-normalised values back into the split DataFrames
    for i, col in enumerate(NIR_COLS):
        df_train[f"SNV_{col}"] = X_train_snv[:, i]
        df_test[f"SNV_{col}"]  = X_test_snv[:, i]

    # Save split CSVs (include raw + SNV columns + metadata)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(DATA_DIR / "train.csv", index=False)
    df_test.to_csv(DATA_DIR  / "test.csv",  index=False)
    print(f"\nSaved: {DATA_DIR / 'train.csv'}")
    print(f"Saved: {DATA_DIR / 'test.csv'}")

    # Save preprocessing artefacts
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(snv, PREP_DIR / "snv_params.pkl")
    joblib.dump(le,  PREP_DIR / "label_encoder.pkl")
    print(f"Saved: {PREP_DIR / 'snv_params.pkl'}")
    print(f"Saved: {PREP_DIR / 'label_encoder.pkl'}")

    print("\nDone. Run 01_train_arch1.py next.\n")


if __name__ == "__main__":
    main()
