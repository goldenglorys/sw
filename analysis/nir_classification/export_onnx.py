#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export trained sklearn models to version-independent formats for Jetson Nano.

  RF, SVM, PCA-LDA  ->  .onnx  (via skl2onnx)
  PLS-DA             ->  .npz   (pure numpy matrix inference)

Run once on the Mac after training:
  source .venv/bin/activate
  pip install skl2onnx onnxruntime
  python analysis/nir_classification/export_onnx.py

The generated files work on any Python version via onnxruntime -- no sklearn
version match required on the Jetson.
"""
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nir_utils import SNVTransformer, PLSDAClassifier  # noqa: F401

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

N_FEATURES = 18
MODEL_DIR  = Path(__file__).resolve().parent / "models"
OPSET      = 12   # widely supported; onnxruntime 1.10+ handles this fine


def to_onnx(name, model, out_path):
    """Convert a sklearn estimator or pipeline to ONNX and save."""
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]

    # zipmap=False: probabilities come out as float array, not list-of-dicts
    final_estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    options = {type(final_estimator): {"zipmap": False}}

    onnx_model = convert_sklearn(
        model, initial_types=initial_type, options=options, target_opset=OPSET
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("  Saved: {}".format(out_path))


def export_plsda_npz(out_path):
    """
    Extract PLSDAClassifier weight matrices for pure numpy inference.

    Inference formula (verified to match sklearn.predict exactly):
      X_pca  = (X - pca_mean) @ pca_components.T
      X_c    = X_pca - pls_x_mean          # center only, no scale
      Y_hat  = X_c @ pls_coef.T + pls_intercept
      pred   = argmax(Y_hat, axis=1)
    """
    clf = joblib.load(MODEL_DIR / "arch2_plsda.pkl")
    pls = clf.pls

    matrices = {
        "pca_mean":       clf.pca.mean_,
        "pca_components": clf.pca.components_,
        "pls_x_mean":     pls._x_mean,
        "pls_coef":       pls.coef_,
        "pls_intercept":  pls.intercept_,
    }
    np.savez(str(out_path), **matrices)
    print("  Saved: {}".format(out_path))

    # Sanity-check against sklearn on a few random samples
    rng = np.random.default_rng(42)
    test_x = rng.standard_normal((20, N_FEATURES))
    sk_pred  = clf.predict(test_x)
    mats     = np.load(str(out_path))
    X_pca    = (test_x - mats["pca_mean"]) @ mats["pca_components"].T
    X_c      = X_pca - mats["pls_x_mean"]
    Y_hat    = X_c @ mats["pls_coef"].T + mats["pls_intercept"]
    np_pred  = np.argmax(Y_hat, axis=1)
    match    = np.sum(sk_pred == np_pred)
    print("  Verification: {}/{} predictions match sklearn".format(match, len(sk_pred)))


def main():
    print("=" * 60)
    print("  Exporting sklearn models to ONNX / numpy format")
    print("  (version-independent -- works on Jetson Python 3.6)")
    print("=" * 60)

    print("\nArch 1 -- Random Forest")
    to_onnx("rf", joblib.load(MODEL_DIR / "arch1_rf.pkl"), MODEL_DIR / "arch1_rf.onnx")

    print("\nArch 1 -- SVM")
    to_onnx("svm", joblib.load(MODEL_DIR / "arch1_svm.pkl"), MODEL_DIR / "arch1_svm.onnx")

    print("\nArch 2 -- PCA-LDA")
    to_onnx("pcalda", joblib.load(MODEL_DIR / "arch2_pcalda.pkl"), MODEL_DIR / "arch2_pcalda.onnx")

    print("\nArch 2 -- PLS-DA  (numpy matrices)")
    export_plsda_npz(MODEL_DIR / "arch2_plsda_matrices.npz")

    print("\nDone. Commit these files and pull on the Jetson, then:")
    print("  pip3 install onnxruntime==1.10.0")
    print("  python3 run_inference.py --model rf svm --material PE --mock")


if __name__ == "__main__":
    main()
