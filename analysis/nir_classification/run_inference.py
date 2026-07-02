#!/usr/bin/env python3
"""
Jetson Nano real-time NIR material inference.

Takes live spectral readings from the SparkFun AS7265x sensor, runs them
through one or more trained classifiers, prints results, and appends every
reading to a CSV that is structurally compatible with the training dataset —
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

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent   # analysis/nir_classification/
ROOT     = THIS_DIR.parents[1]               # project root

sys.path.insert(0, str(THIS_DIR))            # nir_utils
sys.path.insert(0, str(ROOT / "sparkfun"))   # nir_sensor

from nir_utils import SNVTransformer, PLSDAClassifier  # noqa: F401 — pickle needs these

# ── constants ─────────────────────────────────────────────────────────────────
PREP_DIR  = THIS_DIR / "preprocessing"
MODEL_DIR = THIS_DIR / "models"
DEFAULT_OUT = ROOT / "logs" / "csvs" / "inference_results.csv"

NIR_WAVELENGTHS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
]
NIR_COLS = [f"NIR_{wl}nm" for wl in NIR_WAVELENGTHS]

# name → (backend, path)
MODEL_REGISTRY = {
    "rf":     ("sklearn", MODEL_DIR / "arch1_rf.pkl"),
    "svm":    ("sklearn", MODEL_DIR / "arch1_svm.pkl"),
    "plsda":  ("sklearn", MODEL_DIR / "arch2_plsda.pkl"),
    "pcalda": ("sklearn", MODEL_DIR / "arch2_pcalda.pkl"),
    "mlp":    ("keras",   MODEL_DIR / "arch3_mlp.keras"),
    "cnn":    ("keras",   MODEL_DIR / "arch3_cnn.keras"),
}

AUTO_QUIT_SECONDS = 3   # countdown after reads complete (fixed-material mode)


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(keys: list[str]) -> dict:
    """Load and return {name: model} for the requested keys."""
    if "all" in keys:
        keys = list(MODEL_REGISTRY.keys())

    loaded = {}
    keras_needed = any(MODEL_REGISTRY[k][0] == "keras" for k in keys if k in MODEL_REGISTRY)
    if keras_needed:
        import tensorflow as tf  # lazy import — TF startup is slow
        _keras = tf.keras
    else:
        _keras = None

    for name in keys:
        if name not in MODEL_REGISTRY:
            print(f"  WARNING: unknown model '{name}' — skipping. "
                  f"Valid keys: {', '.join(MODEL_REGISTRY)}")
            continue
        backend, path = MODEL_REGISTRY[name]
        if not path.exists():
            print(f"  WARNING: {name} model not found at {path} — skipping.")
            continue
        if backend == "sklearn":
            loaded[name] = ("sklearn", joblib.load(path))
        else:
            loaded[name] = ("keras", _keras.models.load_model(path))
        print(f"  Loaded: {name:8s}  ({backend})")
    return loaded


# ── preprocessing ─────────────────────────────────────────────────────────────

def load_preprocessing():
    snv = joblib.load(PREP_DIR / "snv_params.pkl")
    le  = joblib.load(PREP_DIR / "label_encoder.pkl")
    return snv, le


def preprocess(raw_values: list[float], snv: SNVTransformer) -> np.ndarray:
    """raw 18 floats → SNV-normalised row vector shape (1, 18)."""
    x = np.array(raw_values, dtype=np.float64).reshape(1, -1)
    return snv.transform(x)


# ── inference ─────────────────────────────────────────────────────────────────

def run_models(x_snv: np.ndarray, models: dict, le) -> dict:
    """
    Run all loaded models on a single SNV-normalised row.

    Returns {name: {"pred": str, "conf": float|None}}
    conf is the probability of the predicted class (None if unavailable).
    """
    results = {}
    for name, (backend, model) in models.items():
        if backend == "sklearn":
            idx = int(model.predict(x_snv)[0])

            # confidence: try predict_proba, fall back to decision_function
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(x_snv)[0]
                    conf = float(probs[idx])
                except Exception:
                    pass
            if conf is None and hasattr(model, "decision_function"):
                try:
                    scores = model.decision_function(x_snv)[0]
                    # softmax normalisation of decision scores
                    e = np.exp(scores - scores.max())
                    conf = float(e[idx] / e.sum())
                except Exception:
                    pass

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


# ── sensor ────────────────────────────────────────────────────────────────────

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


def mock_reading() -> dict:
    """Synthetic reading for testing without hardware."""
    np.random.seed(int(time.time() * 1000) % 2**31)
    base = np.random.uniform(50, 800, 18).tolist()
    return {
        "values":      base,
        "temperature": round(np.random.uniform(26, 29), 1),
        "timestamp":   time.time(),
    }


# ── CSV ───────────────────────────────────────────────────────────────────────

def build_headers(model_names: list[str]) -> list[str]:
    base = (
        ["Timestamp", "Material_Type", "Label",
         "Sample_Number", "Session_Scan_Number"]
        + NIR_COLS
        + ["NIR_Temperature"]
    )
    for name in model_names:
        base += [f"pred_{name}", f"conf_{name}"]
    return base


def build_row(
    ts: str, material: str, label: str,
    sample_num: int, session_num: int,
    raw_values: list[float], temperature: float,
    model_results: dict, model_names: list[str],
) -> list:
    row = [ts, material, label, sample_num, session_num] + raw_values + [temperature]
    for name in model_names:
        r = model_results.get(name)
        row += [r["pred"] if r else "", f"{r['conf']:.4f}" if (r and r["conf"] is not None) else ""]
    return row


def open_csv(path: Path, headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(headers)
    return fh, writer


# ── scan loop ─────────────────────────────────────────────────────────────────

def take_batch(
    sensor_or_mock, use_mock: bool, reads: int,
    material: str, label: str,
    sample_num: int, session_num: int,
    models: dict, model_names: list[str],
    snv, le,
    writer, fh,
) -> tuple[int, int]:
    """Fire `reads` sequential scans, run inference, write to CSV. Returns updated counters."""
    for i in range(reads):
        prefix = f"  [{i+1}/{reads}]" if reads > 1 else " "
        print(f"{prefix} Scanning...", end=" ", flush=True)

        data = mock_reading() if use_mock else sensor_or_mock.take_measurement()

        sample_num  += 1
        session_num += 1

        ts = datetime.fromtimestamp(data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        x_snv = preprocess(data["values"], snv)
        results = run_models(x_snv, models, le)

        # console output
        preds_str = "  ".join(
            f"{n}={r['pred']}"
            + (f"({r['conf']*100:.0f}%)" if r["conf"] is not None else "")
            for n, r in results.items()
        )
        peak_idx = int(np.argmax(data["values"]))
        print(
            f"done.  peak={NIR_WAVELENGTHS[peak_idx]}nm  "
            f"temp={data['temperature']:.1f}°C\n"
            f"          {preds_str}"
        )

        row = build_row(
            ts, material, label, sample_num, session_num,
            data["values"], data["temperature"], results, model_names,
        )
        writer.writerow(row)
        fh.flush()

    return sample_num, session_num


def input_with_timeout(prompt: str, timeout: float) -> str | None:
    print(prompt, end="", flush=True)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip().lower()
        print()
        return None
    except (AttributeError, OSError):
        return input().strip().lower()


def scan_loop(
    sensor_or_mock, use_mock: bool, material: str, label: str,
    models: dict, model_names: list[str], snv, le,
    reads: int, writer, fh,
):
    label_str = f", label={label}" if label else ""
    reads_str = f"{reads} reading{'s' if reads > 1 else ''} per trigger"
    print(f"\n[{material}{label_str}]  Enter=scan ({reads_str}) | 'q'=quit\n")

    sample_num  = 0
    session_num = 0
    first = True

    while True:
        if first:
            try:
                cmd = input(f"  [{material}] > ").strip().lower()
            except EOFError:
                break
            first = False
        else:
            cmd = input_with_timeout(
                f"\n  [{material}] > Press Enter to scan again, "
                f"or auto-quitting in {AUTO_QUIT_SECONDS}s... ",
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


# ── main ──────────────────────────────────────────────────────────────────────

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
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data instead of real sensor (for testing off-Jetson)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NIR MATERIAL INFERENCE")
    print("=" * 60)

    # load preprocessing
    try:
        snv, le = load_preprocessing()
    except FileNotFoundError:
        print("ERROR: preprocessing artefacts not found.")
        print("  Run: python analysis/nir_classification/00_split_and_preprocess.py")
        sys.exit(1)
    print(f"Classes: {list(le.classes_)}")

    # load models
    print("\nLoading models...")
    models = load_models(args.model)
    if not models:
        print("ERROR: no models loaded. Check model keys and that training has been run.")
        sys.exit(1)
    model_names = list(models.keys())

    # sensor / mock
    sensor = None
    if args.mock:
        print("\nMock mode — using synthetic spectral data.")
    else:
        print("\nInitialising NIR sensor...", end=" ", flush=True)
        try:
            sensor = init_sensor()
            print("ready.")
            warmup(sensor)
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            print("  Use --mock to test without sensor hardware.")
            sys.exit(1)

    # output CSV
    out_path = Path(args.output)
    headers  = build_headers(model_names)
    fh, writer = open_csv(out_path, headers)

    print(f"\nOutput CSV : {out_path}")
    print(f"Material   : {args.material}")
    if args.label:
        print(f"Label      : {args.label}")
    print(f"Models     : {', '.join(model_names)}")
    print(f"Reads/scan : {args.reads}")
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
    print(f"  Done. {total} reading(s) saved to:")
    print(f"  {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
