# NIR Plastic Classification — Results

## Dataset

| | |
|---|---|
| Features | 18 NIR channels (410nm – 940nm) |
| Target | `Material_Type` (5 classes) |
| Split | 70/30 stratified, seed=42 → **417 train / 180 test** |

**Class distribution:**

| Class | Total | Train | Test |
|---|---|---|---|
| PP | 168 | 117 | 51 |
| PET | 138 | 97 | 41 |
| HDPE | 114 | 80 | 34 |
| PE | 105 | 73 | 32 |
| LDPE | 72 ⚠ | 50 | 22 |

LDPE is under-represented (72 samples) and is the weakest class across all models.

---

## Preprocessing

- **SNV (Standard Normal Variate):** row-wise mean subtraction and std division, applied per spectrum. Corrects for intensity variation due to sensor distance and surface texture.
- Fit on train set only, then applied to test set.
- `NIR_Temperature` excluded (range 26–29°C, std 0.6°C — negligible signal).

---

## Model Architectures

### Arch 1 — Classical ML

**Random Forest**
An ensemble of 200 decision trees. Each tree learns spectral thresholds ("if NIR_610nm > X and NIR_730nm < Y → PET"), final prediction is majority vote across all trees. Robust to noise, works well when features outnumber samples. Tuned via 5-fold CV grid search.

**SVM with RBF kernel**
Finds the widest decision boundary between classes. The RBF kernel implicitly maps the 18 channel values into a much higher-dimensional space where classes become linearly separable. Fast at inference because it only compares a new reading against the small set of "support vectors" retained from training. Also tuned via 5-fold CV grid search.

---

### Arch 2 — Chemometrics

**PCA → PLS-DA**
Compresses 18 channels down to 10 principal components (PCA), then PLS-DA fits a regression from those components to one-hot class labels and predicts by argmax. This is the textbook NIR spectroscopy approach in analytical chemistry labs. Works best when classes are linearly separable after compression — PE and LDPE overlap in PCA space, which is why it struggles here.

**PCA → LDA**
Same PCA compression, but Linear Discriminant Analysis replaces PLS-DA. LDA directly maximises between-class separation while minimising within-class spread, making it a better classifier than PLS-DA for this problem. Still limited by the PCA bottleneck.

---

### Arch 3 — Neural Networks

**MLP (Multi-Layer Perceptron)**
Fully connected network: Input(18) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Softmax(5 classes). Each layer learns weighted combinations of all previous activations. Dropout randomly disables 30% of neurons per batch to reduce memorisation. Trained with Adam (lr=0.001), early stopping patience=15 on validation loss.

**1D CNN (Convolutional Neural Network)**
Treats the spectrum as a 1D signal: Input(18,1) → Conv1D(32 filters, k=3, same) → MaxPool → Conv1D(64 filters, k=3, same) → Flatten → Dense(64) → Dropout(0.3) → Softmax(5). Convolutional filters slide across adjacent wavelengths and detect local spectral shapes (e.g. a peak at 610nm followed by a trough). This gives it an edge over the MLP because NIR spectral features are inherently local — adjacent channels are physically correlated. Same training setup as MLP.

---

## Results

| Model | Accuracy | Macro F1 | Infer (ms/sample) |
|---|---|---|---|
| **Random Forest** | **96.1%** | **0.9545** | 0.15 |
| SVM (RBF) | 95.0% | 0.9435 | **0.008** |
| 1D CNN | 86.1% | 0.8564 | 0.39 |
| MLP | 82.2% | 0.8178 | 0.32 | 
| PCA → LDA | 74.4% | 0.7415 | 0.001 | 
| PCA → PLS-DA | 70.6% | 0.6770 | 0.001 |

### Per-class breakdown (best model — Random Forest)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| HDPE | 0.92 | 1.00 | 0.96 |
| LDPE | 1.00 | 0.86 | 0.93 |
| PE | 0.97 | 0.91 | 0.94 |
| PET | 0.93 | 0.98 | 0.95 |
| PP | 1.00 | 1.00 | **1.00** |

---

## Key Findings

- **PP is trivially separable** — 100% precision and recall across RF and SVM. Distinctive NIR signature.
- **LDPE is the hardest class** in every model. It sits close to PE in PCA space (both are polyethylene). With only 50 training samples, all models show reduced recall here.
- **PLS-DA fails on LDPE** — only 36% recall. PLS-DA assumes linearly separable latent structure; PE/LDPE overlap violates this.
- **Neural networks underperform classical methods** on this dataset size (~83 samples/class average). The CNN (86%) beats the MLP (82%) because local spectral features help even at 18 channels, but both fall well short of RF/SVM.
- **PCA explains 71% of variance in 2 components** (PC1=55%, PC2=16%), confirming the data is low-dimensional and suited to classical ML.

