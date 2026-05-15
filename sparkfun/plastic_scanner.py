#!/usr/bin/env python3
"""
Standalone plastic NIR data collection script.
Scans one material at a time (on-demand) and saves all readings to a single CSV.
No camera required — NIR sensor only.

Usage:
    python sparkfun/plastic_scanner.py
    python sparkfun/plastic_scanner.py --output /path/to/output.csv
"""
import sys
import os
import csv
import argparse
from datetime import datetime
from pathlib import Path

# Works whether run from project root or from sparkfun/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nir_sensor import NIRSensor


PRESET_MATERIALS = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

CSV_HEADERS = (
    ["Timestamp", "Material_Type", "Sample_Number", "Session_Scan_Number"]
    + [f"NIR_{wl}nm" for wl in NIRSensor.WAVELENGTHS]
    + ["NIR_Temperature"]
)


def build_output_path():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "logs" / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"plastic_nir_{timestamp}.csv"


def prompt_material():
    print(f"\nPreset materials: {', '.join(PRESET_MATERIALS)}")
    while True:
        raw = input("Enter material type (or 'q' to finish session): ").strip()
        if raw.lower() == 'q':
            return None
        if raw:
            return raw.upper()
        print("  Material name cannot be empty.")


def scan_loop(sensor, material, start_sample_num, session_count, writer, csv_file):
    """
    Interactive scan loop for one material type.
    Returns (new_session_count, new_sample_num, should_quit).
    """
    sample_num = start_sample_num
    print(f"\n[{material}] Place sample under sensor.")
    print("  Enter = scan | 'n' = next material | 'q' = finish session\n")

    while True:
        try:
            cmd = input(f"  [{material}] > ").strip().lower()
        except EOFError:
            return session_count, sample_num, True

        if cmd == 'q':
            return session_count, sample_num, True
        if cmd == 'n':
            return session_count, sample_num, False
        if cmd != '':
            print("  Enter=scan | n=next material | q=quit")
            continue

        # Blank Enter → take one scan
        print("  Scanning...", end=" ", flush=True)
        try:
            data = sensor.take_measurement()
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        sample_num += 1
        session_count += 1

        iso_ts = datetime.fromtimestamp(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = (
            [iso_ts, material, sample_num, session_count]
            + data['values']
            + [data['temperature']]
        )
        writer.writerow(row)
        csv_file.flush()

        peak_idx = data['values'].index(max(data['values']))
        peak_wl = NIRSensor.WAVELENGTHS[peak_idx]
        print(
            f"done.  "
            f"peak={peak_wl}nm ({data['values'][peak_idx]:.1f}), "
            f"temp={data['temperature']:.1f}°C  "
            f"[reading #{session_count} saved]"
        )


def main(output_path=None):
    if output_path is None:
        output_path = build_output_path()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  PLASTIC NIR SCANNER")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print("=" * 60)

    print("\nInitializing NIR sensor...", end=" ", flush=True)
    try:
        sensor = NIRSensor()
    except Exception as e:
        print(f"\nERROR: Could not initialize sensor: {e}")
        sys.exit(1)
    print("ready.\n")

    material_counts = {}  # material -> number of samples collected
    session_count = 0
    quit_session = False

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADERS)

        try:
            while not quit_session:
                material = prompt_material()
                if material is None:
                    break

                start_sample_num = material_counts.get(material, 0)
                session_count, new_sample_num, quit_session = scan_loop(
                    sensor, material, start_sample_num, session_count, writer, csv_file
                )
                material_counts[material] = new_sample_num

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
        finally:
            sensor.close()

    print("\n" + "=" * 60)
    print("  SESSION COMPLETE")
    print("=" * 60)
    print(f"  Total readings : {session_count}")
    print(f"  Output file    : {output_path}")
    if material_counts:
        print()
        print("  Breakdown:")
        for mat, count in sorted(material_counts.items()):
            label = "reading" if count == 1 else "readings"
            print(f"    {mat:<10} {count} {label}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect NIR spectra for plastic samples — no camera required."
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output CSV path (default: logs/csvs/plastic_nir_<timestamp>.csv)"
    )
    args = parser.parse_args()
    main(output_path=args.output)
