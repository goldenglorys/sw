#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export trained sklearn models to version-independent formats for Jetson Nano.

  RF        -> arch1_rf_trees.npz   (pure numpy tree traversal)
  SVM       -> arch1_svm_numpy.npz  (pure numpy RBF+OVO inference)
  PLS-DA    -> arch2_plsda_matrices.npz
  PCA-LDA   -> arch2_pcalda_matrices.npz

No onnxruntime needed on the Jetson.  All four sklearn models use pure numpy.
MLP and CNN remain as .keras files (need TensorFlow).

Run once on the Mac after training:
  source .venv/bin/activate
  python analysis/nir_classification/export_onnx.py
"""
import sys
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nir_utils import SNVTransformer, PLSDAClassifier  # noqa: F401

N_FEATURES = 18
MODEL_DIR  = Path(__file__).resolve().parent / "models"
DATA_DIR   = Path(__file__).resolve().parent / "data"

SNV_COLS = [
    "SNV_NIR_410nm", "SNV_NIR_435nm", "SNV_NIR_460nm", "SNV_NIR_485nm",
    "SNV_NIR_510nm", "SNV_NIR_535nm", "SNV_NIR_560nm", "SNV_NIR_585nm",
    "SNV_NIR_610nm", "SNV_NIR_645nm", "SNV_NIR_680nm", "SNV_NIR_705nm",
    "SNV_NIR_730nm", "SNV_NIR_760nm", "SNV_NIR_810nm", "SNV_NIR_860nm",
    "SNV_NIR_900nm", "SNV_NIR_940nm",
]


def load_test_X():
    return pd.read_csv(DATA_DIR / "test.csv")[SNV_COLS].values.astype(np.float64)


def export_rf_npz(out_path):
    """Extract RF tree arrays for vectorised numpy traversal."""
    rf = joblib.load(MODEL_DIR / "arch1_rf.pkl")
    n_trees = len(rf.estimators_)
    depths = [t.tree_.node_count for t in rf.estimators_]
    max_nodes = max(depths)

    CL  = np.full((n_trees, max_nodes), -2, dtype=np.intp)
    CR  = np.full((n_trees, max_nodes), -2, dtype=np.intp)
    FT  = np.full((n_trees, max_nodes), -2, dtype=np.intp)
    THR = np.full((n_trees, max_nodes), -2.0, dtype=np.float64)
    VAL = np.zeros((n_trees, max_nodes, rf.n_classes_), dtype=np.float64)

    for t_idx, tree in enumerate(rf.estimators_):
        nc = tree.tree_.node_count
        CL[t_idx, :nc]     = tree.tree_.children_left
        CR[t_idx, :nc]     = tree.tree_.children_right
        FT[t_idx, :nc]     = tree.tree_.feature
        THR[t_idx, :nc]    = tree.tree_.threshold
        VAL[t_idx, :nc, :] = tree.tree_.value[:, 0, :]

    np.savez(str(out_path),
             children_left=CL, children_right=CR,
             feature=FT, threshold=THR, value=VAL)
    print("  Saved: {}".format(out_path))

    # Verify
    X = load_test_X()
    sk_pred = rf.predict(X)
    mats = np.load(str(out_path))
    CL2, CR2, FT2, THR2, VAL2 = (
        mats["children_left"], mats["children_right"],
        mats["feature"], mats["threshold"], mats["value"]
    )
    n_t = CL2.shape[0]
    tidx = np.arange(n_t)

    np_pred = []
    for row in X:
        nodes = np.zeros(n_t, dtype=np.intp)
        for _ in range(200):
            lc = CL2[tidx, nodes]
            if (lc == -1).all():
                break
            go_left = row[FT2[tidx, nodes]] <= THR2[tidx, nodes]
            new_n = np.where(go_left, CL2[tidx, nodes], CR2[tidx, nodes])
            nodes = np.where(lc == -1, nodes, new_n)
        votes = VAL2[tidx, nodes, :].sum(axis=0)
        np_pred.append(int(np.argmax(votes)))

    match = int(np.sum(sk_pred == np.array(np_pred)))
    print("  Verification: {}/{} match sklearn".format(match, len(X)))


def export_svm_npz(out_path):
    """
    Extract SVM RBF parameters for pure numpy OVO inference.

    Decision function for binary classifier (i, j) with i < j:
      K       = exp(-gamma * ||x - sv||^2)  for each support vector
      coef_i  = dual_coef[j-1, sv_range_i]  (class i SVs in this classifier)
      coef_j  = dual_coef[i,   sv_range_j]  (class j SVs in this classifier)
      dec     = dot(coef_i, K[sv_range_i]) + dot(coef_j, K[sv_range_j]) + intercept[clf_idx]
      vote i if dec > 0 else vote j
    Classifiers ordered: (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1)
    """
    svm = joblib.load(MODEL_DIR / "arch1_svm.pkl")
    np.savez(str(out_path),
             support_vectors=svm.support_vectors_,
             dual_coef=svm.dual_coef_,
             intercept=svm.intercept_,
             gamma=np.array([svm._gamma]),
             n_support=svm.n_support_,
             n_classes=np.array([len(svm.classes_)]))
    print("  Saved: {}".format(out_path))

    # Verify
    X = load_test_X()
    sk_pred = svm.predict(X)
    mats = np.load(str(out_path))
    SVs       = mats["support_vectors"]
    dc        = mats["dual_coef"]
    ic        = mats["intercept"]
    gam       = float(mats["gamma"][0])
    n_sup     = mats["n_support"].astype(int)
    n_cls     = int(mats["n_classes"][0])
    sv_start  = np.concatenate([[0], np.cumsum(n_sup[:-1])]).astype(int)

    np_pred = []
    for row in X:
        diff = row - SVs
        K = np.exp(-gam * (diff ** 2).sum(axis=1))
        votes = np.zeros(n_cls)
        clf_idx = 0
        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                i_s, i_e = sv_start[i], sv_start[i] + n_sup[i]
                j_s, j_e = sv_start[j], sv_start[j] + n_sup[j]
                dec = (np.dot(dc[j-1, i_s:i_e], K[i_s:i_e]) +
                       np.dot(dc[i,   j_s:j_e], K[j_s:j_e]) +
                       ic[clf_idx])
                votes[i if dec > 0 else j] += 1
                clf_idx += 1
        np_pred.append(int(np.argmax(votes)))

    match = int(np.sum(sk_pred == np.array(np_pred)))
    print("  Verification: {}/{} match sklearn".format(match, len(X)))


def export_plsda_npz(out_path):
    """
    Extract PLS-DA weight matrices for pure numpy inference.

    Inference formula:
      X_pca = (X - pca_mean) @ pca_components.T
      X_c   = X_pca - pls_x_mean
      Y_hat = X_c @ pls_coef.T + pls_intercept
      pred  = argmax(Y_hat, axis=1)
    """
    clf = joblib.load(MODEL_DIR / "arch2_plsda.pkl")
    pls = clf.pls
    np.savez(str(out_path),
             pca_mean=clf.pca.mean_,
             pca_components=clf.pca.components_,
             pls_x_mean=pls._x_mean,
             pls_coef=pls.coef_,
             pls_intercept=pls.intercept_)
    print("  Saved: {}".format(out_path))

    X = load_test_X()
    sk_pred = clf.predict(X)
    mats    = np.load(str(out_path))
    X_pca   = (X - mats["pca_mean"]) @ mats["pca_components"].T
    X_c     = X_pca - mats["pls_x_mean"]
    Y_hat   = X_c @ mats["pls_coef"].T + mats["pls_intercept"]
    np_pred = np.argmax(Y_hat, axis=1)
    match   = int(np.sum(sk_pred == np_pred))
    print("  Verification: {}/{} match sklearn".format(match, len(X)))


def export_pcalda_npz(out_path):
    """
    Extract PCA-LDA weight matrices for pure numpy inference.

    Inference formula:
      X_pca = (X - pca_mean) @ pca_components.T
      dec   = X_pca @ lda_coef.T + lda_intercept
      pred  = argmax(dec, axis=1)
    """
    pipe = joblib.load(MODEL_DIR / "arch2_pcalda.pkl")
    pca  = pipe["pca"]
    lda  = pipe["lda"]
    np.savez(str(out_path),
             pca_mean=pca.mean_,
             pca_components=pca.components_,
             lda_coef=lda.coef_,
             lda_intercept=lda.intercept_)
    print("  Saved: {}".format(out_path))

    X = load_test_X()
    sk_pred = pipe.predict(X)
    mats    = np.load(str(out_path))
    X_pca   = (X - mats["pca_mean"]) @ mats["pca_components"].T
    dec     = X_pca @ mats["lda_coef"].T + mats["lda_intercept"]
    np_pred = np.argmax(dec, axis=1)
    match   = int(np.sum(sk_pred == np_pred))
    print("  Verification: {}/{} match sklearn".format(match, len(X)))


def main():
    print("=" * 60)
    print("  Exporting sklearn models to pure numpy format")
    print("  (version-independent -- works on Jetson Python 3.6)")
    print("=" * 60)

    print("\nArch 1 -- Random Forest  ->  arch1_rf_trees.npz")
    export_rf_npz(MODEL_DIR / "arch1_rf_trees.npz")

    print("\nArch 1 -- SVM            ->  arch1_svm_numpy.npz")
    export_svm_npz(MODEL_DIR / "arch1_svm_numpy.npz")

    print("\nArch 2 -- PLS-DA         ->  arch2_plsda_matrices.npz")
    export_plsda_npz(MODEL_DIR / "arch2_plsda_matrices.npz")

    print("\nArch 2 -- PCA-LDA        ->  arch2_pcalda_matrices.npz")
    export_pcalda_npz(MODEL_DIR / "arch2_pcalda_matrices.npz")

    print("\nDone. Commit the .npz files and pull on the Jetson:")
    print("  git add analysis/nir_classification/models/*.npz")
    print("  git commit -m 'export: add numpy inference arrays for all 4 sklearn models'")
    print("  # on Jetson:")
    print("  git pull")
    print("  python3 analysis/nir_classification/run_inference.py --model rf svm plsda pcalda --mock")


if __name__ == "__main__":
    main()
