# NIR Plastic Classification — ML Pipeline

Complete guide to training, evaluating, and deploying the NIR spectral classifier
for the reverse vending machine. Run every command from the **project root**.

---

## Prerequisites

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Install dependencies (one-time)

```bash
pip install -r requirements.txt
```

New packages added for this pipeline: `scikit-learn`, `pandas`, `matplotlib`,
`seaborn`, `scipy`, `joblib`.

---

## Dataset

**Location:** `data/nir/plastic_nir_dataset.csv`

| Property | Value |
|---|---|
| Total samples | 597 |
| Material types | 5 (HDPE, LDPE, PE, PET, PP) |
| Feature columns | 18 NIR channels: `NIR_410nm` … `NIR_940nm` |
| Target column | `Material_Type` |
| Metadata (excluded) | `Timestamp`, `Label`, `Sample_Number`, `Session_Scan_Number`, `NIR_Temperature` |

**Class distribution:**

| Material | Samples | Train (70%) | Test (30%) |
|---|---|---|---|
| HDPE | 114 | 80 | 34 |
| LDPE | 72 | 50 | 22 |
| PE | 105 | 73 | 32 |
| PET | 138 | 97 | 41 |
| PP | 168 | 117 | 51 |

> **Note:** LDPE has only 72 samples. Expect slightly weaker recall for this class.
> PE and LDPE are chemically similar — the confusion matrix may show some cross-confusion.

---

## Step 0 — Data Splitting & Preprocessing (already done)

This step was run automatically. You only need to re-run it if you add new data to
`data/nir/plastic_nir_dataset.csv` or want to change the random seed.

```bash
python analysis/nir_classification/00_split_and_preprocess.py
```

**What it does:**
- Loads the CSV (skips the blank first row)
- Stratified 70/30 train/test split with `random_state=42`
- Fits SNV (Standard Normal Variate) normalisation on the training set
- Saves split files and preprocessing artefacts

**Outputs:**

| File | Description |
|---|---|
| `analysis/nir_classification/data/train.csv` | 417 training samples (raw + SNV columns) |
| `analysis/nir_classification/data/test.csv` | 180 test samples (raw + SNV columns) |
| `analysis/nir_classification/preprocessing/snv_params.pkl` | SNV transformer |
| `analysis/nir_classification/preprocessing/label_encoder.pkl` | Class index ↔ material name mapping |

**Expected output:**
```
Loaded 597 samples, 5 material types
Split: 417 train / 180 test (seed=42)
Per-class split:
  HDPE  : train=  80, test=  34
  LDPE  : train=  50, test=  22
  PE    : train=  73, test=  32
  PET   : train=  97, test=  41
  PP    : train= 117, test=  51
```

---

## Step 1 — Architecture 1: Random Forest + SVM

```bash
python analysis/nir_classification/01_train_arch1.py
```

**Optional — with Savitzky-Golay 1st derivative on top of SNV:**
```bash
python analysis/nir_classification/01_train_arch1.py --savgol
```

**What it does:**
- Loads `data/train.csv` and `data/test.csv`
- Random Forest: GridSearchCV (5-fold stratified CV) over:
  - `n_estimators`: 100, 200, 500
  - `max_depth`: None, 5, 10, 20
  - `max_features`: sqrt, log2
- SVM (RBF kernel): GridSearchCV over:
  - `C`: 0.1, 1, 10, 100, 1000
  - `gamma`: scale, auto, 0.001, 0.01, 0.1
- Evaluates best configs on the held-out test set

**Expected runtime:** ~2–5 min for RF, ~3–8 min for SVM (on a laptop CPU).

**Outputs:**

| File | Description |
|---|---|
| `models/arch1_rf.pkl` | Best Random Forest model |
| `models/arch1_svm.pkl` | Best SVM model |
| `plots/confusion_arch1_rf.png` | Confusion matrix heatmap — RF |
| `plots/confusion_arch1_svm.png` | Confusion matrix heatmap — SVM |
| `results/classification_reports.txt` | Appended with RF and SVM reports |

**Expected console output (example — actual numbers will vary):**
```
--- Random Forest ---
  Best params: {'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 200}
  Best CV accuracy: 0.9xxx
  Test accuracy: 0.9xxx

--- SVM (RBF kernel) ---
  Best params: {'C': 10, 'gamma': 'scale'}
  Best CV accuracy: 0.9xxx
  Test accuracy: 0.9xxx
```

---

## Step 2 — Architecture 2: Chemometrics (PLS-DA + PCA→LDA)

```bash
python analysis/nir_classification/02_train_arch2.py
```

**Optional — with Savitzky-Golay derivative:**
```bash
python analysis/nir_classification/02_train_arch2.py --savgol
```

**What it does:**
- Generates a PCA scatter plot (first 2 PCs, all 597 samples, coloured by material)
- Selects best PCA `n_components` (2–10) by 5-fold CV accuracy with LDA
- Trains PCA → PLS-DA (one-hot targets + argmax prediction)
- Trains PCA → LDA as a variant

**Expected runtime:** < 1 minute.

**Outputs:**

| File | Description |
|---|---|
| `models/arch2_plsda.pkl` | PCA → PLS-DA model |
| `models/arch2_pcalda.pkl` | PCA → LDA pipeline |
| `plots/pca_scatter.png` | PCA scatter plot (visual sanity check) |
| `plots/confusion_arch2_plsda.png` | Confusion matrix — PLS-DA |
| `plots/confusion_arch2_pcalda.png` | Confusion matrix — PCA-LDA |
| `results/classification_reports.txt` | Appended |

**Expected console output (example):**
```
Selecting best PCA n_components (2–10) via cross-validation...
    n_components=2: CV accuracy=0.8xxx
    ...
    n_components=6: CV accuracy=0.9xxx  ← example best
  Best n_components: 6

--- PCA → PLS-DA ---
  Test accuracy: 0.9xxx

--- PCA → LDA ---
  Test accuracy: 0.9xxx
```

**Interpreting the PCA scatter plot (`pca_scatter.png`):**
- Well-separated, non-overlapping clusters = clean data, classification should be straightforward
- Overlapping PE/LDPE clusters = expected (chemically similar)

---

## Step 3 — Architecture 3: Neural Networks (MLP + 1D CNN)

```bash
python analysis/nir_classification/03_train_arch3.py
```

**Options:**
```bash
# With Savitzky-Golay derivative
python analysis/nir_classification/03_train_arch3.py --savgol

# Custom max epochs (default 300; early stopping cuts it short)
python analysis/nir_classification/03_train_arch3.py --epochs 500
```

**What it does:**

**MLP:**
- Tries two hidden-layer configs: `[64, 32]` and `[128, 64, 32]`
- Dropout 0.3, Adam lr=1e-3, early stopping patience=15
- 15% validation split from training set
- Saves the best-performing config

**1D CNN:**
- Input shape: `(18, 1)`
- Conv1D(32, k=3, same) → MaxPool1D(2) → Conv1D(64, k=3, same) → Flatten → Dense(64) → Dropout(0.3) → softmax
- Same training setup as MLP

**Expected runtime:** 1–5 min each (early stopping usually fires before 300 epochs with this dataset size).

**Outputs:**

| File | Description |
|---|---|
| `models/arch3_mlp.keras` | Best MLP model |
| `models/arch3_cnn.keras` | 1D CNN model |
| `plots/training_curves_mlp.png` | Loss + accuracy curves per epoch — MLP |
| `plots/training_curves_cnn.png` | Loss + accuracy curves per epoch — CNN |
| `plots/confusion_arch3_mlp.png` | Confusion matrix — MLP |
| `plots/confusion_arch3_cnn.png` | Confusion matrix — CNN |
| `results/classification_reports.txt` | Appended |

**Expected console output:**
```
GPU available: False   (or True on Jetson)

--- MLP ---
  Config: hidden layers = [64, 32]
  Epochs: 87  Test acc: 0.8xxx
  Config: hidden layers = [128, 64, 32]
  Epochs: 102  Test acc: 0.8xxx
  Best MLP config: [64, 32]

--- 1D CNN ---
  Epoch 1/300 ...
  Early stopping at epoch ~120
  Test accuracy: 0.8xxx
```

> **Note:** Neural networks are expected to underperform classical methods on this dataset
> (~415 training samples / 5 classes). This is normal and expected — the comparison table
> will confirm it. See the production recommendation below.

---

## Step 4 — Evaluate All Models (Comparison Table)

Run this **after** all three training steps complete.

```bash
python analysis/nir_classification/04_evaluate_all.py
```

**What it does:**
- Loads every saved model
- Re-runs inference on the 180-sample test set
- Prints per-class precision/recall/F1 for each model
- Prints a side-by-side comparison table
- Highlights best model per metric (accuracy, macro-F1, inference speed)

**Outputs:**

| File | Description |
|---|---|
| `results/comparison_table.csv` | Summary table (all models) |
| `results/classification_reports.txt` | Final comparison appended |

**Expected console output:**
```
======================================================================
  COMPARISON TABLE
======================================================================
      Model  Accuracy  Macro F1  Infer (ms/sample)
  Arch1 RF     0.9xxx    0.9xxx             0.xxxx
 Arch1 SVM     0.9xxx    0.9xxx             0.xxxx
Arch2 PLS-DA   0.8xxx    0.8xxx             0.xxxx
Arch2 PCA-LDA  0.9xxx    0.8xxx             0.xxxx
  Arch3 MLP    0.8xxx    0.8xxx             0.xxxx
  Arch3 CNN    0.8xxx    0.8xxx             0.xxxx

  Best accuracy : Arch1 RF  (or SVM)
  Best macro-F1 : Arch1 RF  (or SVM)
  Fastest infer : Arch1 RF
======================================================================
```

---

## Running with Savitzky-Golay Derivative (Second Pass)

Add `--savgol` to any of steps 1–3 to apply a 1st-order Savitzky-Golay derivative
(window=7, poly=2) on top of the SNV-normalised spectra.

This can improve accuracy when baseline drift between samples is present, at the
cost of slightly more preprocessing. Compare accuracy with and without it.

```bash
python analysis/nir_classification/01_train_arch1.py --savgol
python analysis/nir_classification/02_train_arch2.py --savgol
python analysis/nir_classification/03_train_arch3.py --savgol
```

---

## All Outputs at a Glance

```
analysis/nir_classification/
├── data/
│   ├── train.csv                    417 samples, raw + SNV columns
│   └── test.csv                     180 samples, raw + SNV columns
├── preprocessing/
│   ├── snv_params.pkl               SNV transformer (fitted on train set)
│   └── label_encoder.pkl            {0: HDPE, 1: LDPE, 2: PE, 3: PET, 4: PP}
├── models/
│   ├── arch1_rf.pkl                 Random Forest (best hyperparams)
│   ├── arch1_svm.pkl                SVM RBF (best hyperparams)
│   ├── arch2_plsda.pkl              PCA → PLS-DA
│   ├── arch2_pcalda.pkl             PCA → LDA
│   ├── arch3_mlp.keras              MLP (best config)
│   └── arch3_cnn.keras              1D CNN
├── plots/
│   ├── pca_scatter.png              PCA scatter — all 597 samples
│   ├── confusion_arch1_rf.png
│   ├── confusion_arch1_svm.png
│   ├── confusion_arch2_plsda.png
│   ├── confusion_arch2_pcalda.png
│   ├── confusion_arch3_mlp.png
│   ├── confusion_arch3_cnn.png
│   ├── training_curves_mlp.png
│   └── training_curves_cnn.png
└── results/
    ├── classification_reports.txt   Full per-class reports (all models, appended)
    └── comparison_table.csv         Side-by-side summary
```

---

## Recommended Production Model

Based on the problem characteristics (597 samples, 18 features, 5 classes, edge CPU):

| Recommendation | Why |
|---|---|
| **SVM (RBF)** as first choice | Best accuracy per sample on small spectral datasets; inference < 1ms; no GPU needed |
| **Random Forest** as fallback | Slightly lower accuracy, easier to interpret feature importance |
| **Avoid neural networks in production** until you have 150+ samples per class | With 50 LDPE train samples, NNs will overfit or underfit |

To use the best model in production:
```python
import joblib, numpy as np

snv   = joblib.load("analysis/nir_classification/preprocessing/snv_params.pkl")
le    = joblib.load("analysis/nir_classification/preprocessing/label_encoder.pkl")
model = joblib.load("analysis/nir_classification/models/arch1_svm.pkl")

# spectrum = np.array([...18 NIR channel values...])
spectrum_snv = snv.transform(spectrum.reshape(1, -1))
label_idx    = model.predict(spectrum_snv)[0]
material     = le.inverse_transform([label_idx])[0]
print(material)   # e.g. "PET"
```

---

## Step 5 — Live Inference on Jetson Nano

```bash
python analysis/nir_classification/run_inference.py --model all
```

**All flags:**

| Flag | Short | Default | Description |
| --- | --- | --- | --- |
| `--model` | `-M` | `all` | One or more of: `rf svm plsda pcalda mlp cnn all` |
| `--material` | `-m` | `UNKNOWN` | Ground-truth material type (stored in CSV for later eval) |
| `--label` | `-l` | _(blank)_ | Free-text label e.g. `clean_bottle` |
| `--reads` | `-r` | `1` | Readings per Enter press |
| `--output` | `-o` | `logs/csvs/inference_results.csv` | CSV path |
| `--mock` | | | Synthetic data — runs without sensor hardware |

**Examples:**

```bash
# All models, unknown material (just exploring)
python analysis/nir_classification/run_inference.py --model all

# RF and SVM only, ground truth supplied, 3 reads per trigger
python analysis/nir_classification/run_inference.py --model rf svm -m PET -l clean -r 3

# Test on laptop without sensor
python analysis/nir_classification/run_inference.py --model all --mock -m HDPE

# Custom output file
python analysis/nir_classification/run_inference.py --model all -m PP -o /data/field_run1.csv
```

**What the console shows per reading:**

```text
  [1/2] Scanning... done.  peak=730nm  temp=28.1°C
          rf=PET(94%)  svm=PET(97%)  pcalda=PET(100%)  mlp=PET(88%)  cnn=PET(91%)
```

**Output CSV schema:**

The CSV is structurally identical to the training dataset — same `NIR_*` columns in the same order — with prediction columns appended for each loaded model:

```text
Timestamp, Material_Type, Label, Sample_Number, Session_Scan_Number,
NIR_410nm … NIR_940nm, NIR_Temperature,
pred_rf, conf_rf,
pred_svm, conf_svm,
pred_plsda, conf_plsda,
pred_pcalda, conf_pcalda,
pred_mlp, conf_mlp,
pred_cnn, conf_cnn
```

`conf_*` is the probability of the predicted class (0–1). For SVM this comes from normalising the decision function scores; for all other models it is the softmax/predict_proba output.

**Running a confusion matrix on field results later:**

```python
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("logs/csvs/inference_results.csv")
df = df[df["Material_Type"] != "UNKNOWN"]   # keep only labelled rows

cm = confusion_matrix(df["Material_Type"], df["pred_rf"], labels=["HDPE","LDPE","PE","PET","PP"])
ConfusionMatrixDisplay(cm, display_labels=["HDPE","LDPE","PE","PET","PP"]).plot()
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'joblib'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: train.csv` | Run `python analysis/nir_classification/00_split_and_preprocess.py` first |
| `ModuleNotFoundError: No module named 'tensorflow'` | TF is already in requirements.txt; ensure `.venv` is active |
| SVM grid search hangs | It's running — 5×5×5=125 combinations × 5 folds = 625 fits. Normal on first run. |
| CNN training produces NaN loss | Rare with SNV-normalised data. Try `--epochs 100` with a lower learning rate. |
| Low LDPE recall | Expected — only 50 training samples. Collect more LDPE data to fix. |
