#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano real-time NIR material inference.

Takes live spectral readings from the SparkFun AS7265x sensor, runs them
through one or more trained classifiers, prints results, and appends every
reading to a CSV that is structurally compatible with the training dataset --
so confusion matrix analysis later is just a matter of comparing the
Material_Type column against the pred_* columns.

Usage:
  # All models, no ground truth label (just exploring)
  python analysis/nir_classification/run_inference.py --model all

  # Specific models, supply ground truth for later evaluation
  python analysis/nir_classification/run_inference.py --model rf svm --material PET --label clean

  # 3 readings per trigger, all models
  python analysis/nir_classification/run_inference.py --model all -m HDPE -l dirty_cap -r 3

  # Test without sensor (generates synthetic spectra)
  python analysis/nir_classification/run_inference.py --model all --mock

  # Custom output path
  python analysis/nir_classification/run_inference.py --model all -o /data/field_results.csv

Available model keys: rf, svm, plsda, pcalda, mlp, cnn, all
"""
import sys
import csv
import time
import select
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

# -- path setup ----------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent   # analysis/nir_classification/
ROOT     = THIS_DIR.parents[1]               # project root

sys.path.insert(0, str(THIS_DIR))            # nir_utils
sys.path.insert(0, str(ROOT / "sparkfun"))   # nir_sensor

from nir_utils import SNVTransformer, PLSDAClassifier  # noqa: F401 -- pickle needs these

# -- constants -----------------------------------------------------------------
PREP_DIR  = THIS_DIR / "preprocessing"
MODEL_DIR = THIS_DIR / "models"
DEFAULT_OUT = ROOT / "logs" / "csvs" / "inference_results.csv"

NIR_WAVELENGTHS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
]
NIR_COLS = ["NIR_{}nm".format(wl) for wl in NIR_WAVELENGTHS]

# name -> (backend, path)
# rf_npz    : RF tree arrays for pure numpy traversal (no runtime dep)
# plsda_npz : PLS-DA weight matrices for pure numpy inference (no runtime dep)
# onnx      : ONNX model via onnxruntime (SVM, PCA-LDA) -- needs pip3 install onnxruntime==1.10.0
# keras     : TensorFlow SavedModel (MLP, CNN)
MODEL_REGISTRY = {
    "rf":     ("rf_npz",    MODEL_DIR / "arch1_rf_trees.npz"),
    "svm":    ("onnx",      MODEL_DIR / "arch1_svm.onnx"),
    "plsda":  ("plsda_npz", MODEL_DIR / "arch2_plsda_matrices.npz"),
    "pcalda": ("onnx",      MODEL_DIR / "arch2_pcalda.onnx"),
    "mlp":    ("keras",     MODEL_DIR / "arch3_mlp.keras"),
    "cnn":    ("keras",     MODEL_DIR / "arch3_cnn.keras"),
}

AUTO_QUIT_SECONDS = 3   # countdown after reads complete (fixed-material mode)


# -- model loading -------------------------------------------------------------

def load_models(keys):
    """Load and return {name: model} for the requested keys."""
    if "all" in keys:
        keys = list(MODEL_REGISTRY.keys())

    loaded = {}

    onnx_needed  = any(MODEL_REGISTRY[k][0] == "onnx"  for k in keys if k in MODEL_REGISTRY)
    keras_needed = any(MODEL_REGISTRY[k][0] == "keras" for k in keys if k in MODEL_REGISTRY)

    if onnx_needed:
        try:
            import onnxruntime as ort
        except ImportError:
            print("ERROR: onnxruntime not installed.")
            print("  Models needing it (svm, pcalda) will be skipped.")
            print("  To install: pip3 install onnxruntime==1.10.0")
            ort = None
    else:
        ort = None

    if keras_needed:
        import tensorflow as tf  # lazy import - TF startup is slow
        _keras = tf.keras
    else:
        _keras = None

    for name in keys:
        if name not in MODEL_REGISTRY:
            print("  WARNING: unknown model '{}' -- skipping. "
                  "Valid keys: {}".format(name, ', '.join(MODEL_REGISTRY)))
            continue
        backend, path = MODEL_REGISTRY[name]
        if not path.exists():
            print("  WARNING: {} model not found at {} -- skipping.".format(name, path))
            continue
        if backend == "onnx":
            if ort is None:
                print("  SKIP: {} needs onnxruntime (not installed).".format(name))
                continue
            loaded[name] = ("onnx", ort.InferenceSession(str(path)))
        elif backend in ("rf_npz", "plsda_npz"):
            loaded[name] = (backend, np.load(str(path)))
        else:
            loaded[name] = ("keras", _keras.models.load_model(path))
        print("  Loaded: {:8s}  ({})".format(name, backend))
    return loaded


# -- preprocessing -------------------------------------------------------------

def load_preprocessing():
    snv = joblib.load(PREP_DIR / "snv_params.pkl")
    le  = joblib.load(PREP_DIR / "label_encoder.pkl")
    return snv, le


def preprocess(raw_values, snv):
    """raw 18 floats -> SNV-normalised row vector shape (1, 18)."""
    x = np.array(raw_values, dtype=np.float64).reshape(1, -1)
    return snv.transform(x)


# -- inference -----------------------------------------------------------------

def run_models(x_snv, models, le):
    """
    Run all loaded models on a single SNV-normalised row.

    Returns {name: {"pred": str, "conf": float or None}}
    conf is the probability of the predicted class (None if unavailable).
    """
    results = {}
    for name, (backend, model) in models.items():
        if backend == "onnx":
            inp  = {model.get_inputs()[0].name: x_snv.astype(np.float32)}
            out  = model.run(None, inp)
            idx  = int(out[0][0])
            conf = None
            if len(out) > 1 and out[1] is not None:
                probs = np.array(out[1][0])
                conf  = float(probs[idx])

        elif backend == "rf_npz":
            # Random Forest: vectorised tree traversal across all 200 trees
            mats     = model
            CL       = mats["children_left"]
            CR       = mats["children_right"]
            FT       = mats["feature"]
            THR      = mats["threshold"]
            VAL      = mats["value"]
            n_trees  = CL.shape[0]
            tidx     = np.arange(n_trees)
            nodes    = np.zeros(n_trees, dtype=np.intp)
            x        = x_snv[0]
            for _ in range(20):            # bound by max tree depth (actual: 15)
                lc      = CL[tidx, nodes]
                is_leaf = lc == -1
                if is_leaf.all():
                    break
                go_left = x[FT[tidx, nodes]] <= THR[tidx, nodes]
                new_n   = np.where(go_left, CL[tidx, nodes], CR[tidx, nodes])
                nodes   = np.where(is_leaf, nodes, new_n)
            votes = VAL[tidx, nodes, :].sum(axis=0)
            idx   = int(np.argmax(votes))
            conf  = None

        elif backend == "plsda_npz":
            # PLS-DA: pure numpy inference from extracted weight matrices
            mats  = model
            X_pca = (x_snv - mats["pca_mean"]) @ mats["pca_components"].T
            X_c   = X_pca - mats["pls_x_mean"]
            Y_hat = X_c @ mats["pls_coef"].T + mats["pls_intercept"]
            idx   = int(np.argmax(Y_hat, axis=1)[0])
            conf  = None

        else:  # keras
            x_in = x_snv.astype(np.float32)
            if name == "cnn":
                x_in = x_in.reshape(1, 18, 1)
            probs = model.predict(x_in, verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])

        results[name] = {
            "pred": le.inverse_transform([idx])[0],
            "conf": conf,
        }
    return results


# -- sensor --------------------------------------------------------------------

def init_sensor():
    try:
        from nir_sensor import NIRSensor
        sensor = NIRSensor()
        return sensor
    except ImportError:
        raise RuntimeError(
            "sparkfun-qwiic-as7265x not installed. "
            "On Jetson: pip install sparkfun-qwiic-as7265x"
        )
    except ConnectionError as e:
        raise RuntimeError(str(e))


def warmup(sensor):
    print("  Warming up sensor...", end=" ", flush=True)
    try:
        sensor.take_measurement()
    except Exception:
        pass
    print("done.")


def mock_reading():
    """Synthetic reading for testing without hardware."""
    np.random.seed(int(time.time() * 1000) % 2**31)
    base = np.random.uniform(50, 800, 18).tolist()
    return {
        "values":      base,
        "temperature": round(np.random.uniform(26, 29), 1),
        "timestamp":   time.time(),
    }


# -- CSV -----------------------------------------------------------------------

def build_headers(model_names):
    base = (
        ["Timestamp", "Material_Type", "Label",
         "Sample_Number", "Session_Scan_Number"]
        + NIR_COLS
        + ["NIR_Temperature"]
    )
    for name in model_names:
        base += ["pred_{}".format(name), "conf_{}".format(name)]
    return base


def build_row(ts, material, label, sample_num, session_num,
              raw_values, temperature, model_results, model_names):
    row = [ts, material, label, sample_num, session_num] + raw_values + [temperature]
    for name in model_names:
        r = model_results.get(name)
        row += [
            r["pred"] if r else "",
            "{:.4f}".format(r["conf"]) if (r and r["conf"] is not None) else "",
        ]
    return row


def open_csv(path, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(str(path), "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(headers)
    return fh, writer


# -- scan loop -----------------------------------------------------------------

def take_batch(sensor_or_mock, use_mock, reads,
               material, label, sample_num, session_num,
               models, model_names, snv, le, writer, fh):
    """Fire `reads` sequential scans, run inference, write to CSV. Returns updated counters."""
    for i in range(reads):
        prefix = "  [{}/{}]".format(i + 1, reads) if reads > 1 else " "
        print("{} Scanning...".format(prefix), end=" ", flush=True)

        data = mock_reading() if use_mock else sensor_or_mock.take_measurement()

        sample_num  += 1
        session_num += 1

        ts = datetime.fromtimestamp(data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        x_snv = preprocess(data["values"], snv)
        results = run_models(x_snv, models, le)

        preds_parts = []
        for n, r in results.items():
            conf_str = "({:.0f}%)".format(r["conf"] * 100) if r["conf"] is not None else ""
            preds_parts.append("{}={}{}".format(n, r["pred"], conf_str))
        preds_str = "  ".join(preds_parts)

        peak_idx = int(np.argmax(data["values"]))
        print("done.  peak={}nm  temp={:.1f}C".format(
            NIR_WAVELENGTHS[peak_idx], data["temperature"]
        ))
        print("          {}".format(preds_str))

        row = build_row(
            ts, material, label, sample_num, session_num,
            data["values"], data["temperature"], results, model_names,
        )
        writer.writerow(row)
        fh.flush()

    return sample_num, session_num


def input_with_timeout(prompt, timeout):
    print(prompt, end="", flush=True)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip().lower()
        print()
        return None
    except (AttributeError, OSError):
        return input().strip().lower()


def scan_loop(sensor_or_mock, use_mock, material, label,
              models, model_names, snv, le, reads, writer, fh):
    label_str = ", label={}".format(label) if label else ""
    reads_str = "{} reading{} per trigger".format(reads, "s" if reads > 1 else "")
    print("\n[{}{}]  Enter=scan ({}) | 'q'=quit\n".format(material, label_str, reads_str))

    sample_num  = 0
    session_num = 0
    first = True

    while True:
        if first:
            try:
                cmd = input("  [{}] > ".format(material)).strip().lower()
            except EOFError:
                break
            first = False
        else:
            cmd = input_with_timeout(
                "\n  [{}] > Press Enter to scan again, "
                "or auto-quitting in {}s... ".format(material, AUTO_QUIT_SECONDS),
                AUTO_QUIT_SECONDS,
            )
            if cmd is None:
                break

        if cmd == "q":
            break
        if cmd != "":
            print("  Enter=scan | q=quit")
            continue

        sample_num, session_num = take_batch(
            sensor_or_mock, use_mock, reads,
            material, label, sample_num, session_num,
            models, model_names, snv, le, writer, fh,
        )

    return session_num


# -- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time NIR material classification on Jetson Nano.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run_inference.py --model all\n"
            "  python run_inference.py --model rf svm -m PET -l clean\n"
            "  python run_inference.py --model all -m HDPE -r 3\n"
            "  python run_inference.py --model all --mock     # no sensor needed\n"
        ),
    )
    parser.add_argument(
        "--model", "-M", nargs="+", default=["all"],
        metavar="NAME",
        help="Model(s) to run: rf, svm, plsda, pcalda, mlp, cnn, all  (default: all)",
    )
    parser.add_argument("--material", "-m", metavar="TYPE", default="UNKNOWN",
                        help="Ground-truth material type (e.g. PET). Stored in CSV for later eval.")
    parser.add_argument("--label", "-l", metavar="LABEL", default="",
                        help="Free-text label (e.g. clean_bottle). Stored in CSV.")
    parser.add_argument("--reads", "-r", type=int, default=1, metavar="N",
                        help="Number of readings per Enter press (default: 1)")
    parser.add_argument("--output", "-o", metavar="FILE", default=str(DEFAULT_OUT),
                        help="Output CSV path (default: logs/csvs/inference_results.csv)")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data instead of real sensor (for testing off-Jetson)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NIR MATERIAL INFERENCE")
    print("=" * 60)

    try:
        snv, le = load_preprocessing()
    except FileNotFoundError:
        print("ERROR: preprocessing artefacts not found.")
        print("  Run: python analysis/nir_classification/00_split_and_preprocess.py")
        sys.exit(1)
    print("Classes: {}".format(list(le.classes_)))

    print("\nLoading models...")
    models = load_models(args.model)
    if not models:
        print("ERROR: no models loaded. Check model keys and that training has been run.")
        sys.exit(1)
    model_names = list(models.keys())

    sensor = None
    if args.mock:
        print("\nMock mode -- using synthetic spectral data.")
    else:
        print("\nInitialising NIR sensor...", end=" ", flush=True)
        try:
            sensor = init_sensor()
            print("ready.")
            warmup(sensor)
        except RuntimeError as e:
            print("\nERROR: {}".format(e))
            print("  Use --mock to test without sensor hardware.")
            sys.exit(1)

    out_path = Path(args.output)
    headers  = build_headers(model_names)
    fh, writer = open_csv(out_path, headers)

    print("\nOutput CSV : {}".format(out_path))
    print("Material   : {}".format(args.material))
    if args.label:
        print("Label      : {}".format(args.label))
    print("Models     : {}".format(', '.join(model_names)))
    print("Reads/scan : {}".format(args.reads))
    print("=" * 60)

    total = 0
    try:
        total = scan_loop(
            sensor if not args.mock else None,
            args.mock,
            args.material.upper(), args.label,
            models, model_names, snv, le,
            args.reads, writer, fh,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        fh.close()
        if sensor:
            sensor.close()

    print("\n" + "=" * 60)
    print("  Done. {} reading(s) saved to:".format(total))
    print("  {}".format(out_path))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
