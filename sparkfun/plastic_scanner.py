#!/usr/bin/env python3
"""
Standalone plastic NIR data collection script.
Scans one material at a time (on-demand) and appends all readings to a
single persistent CSV. No camera required — NIR sensor only.

Usage:
    # Interactive — prompts for material type each session
    python sparkfun/plastic_scanner.py

    # Non-interactive — material supplied via flag, go straight to scanning
    python sparkfun/plastic_scanner.py --material HDPE
    python sparkfun/plastic_scanner.py -m PET

    # Take 3 readings per Enter press
    python sparkfun/plastic_scanner.py -m HDPE --reads 3

    # Custom CSV path (still appends if file exists)
    python sparkfun/plastic_scanner.py -m HDPE --output /path/to/dataset.csv
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
DEFAULT_CSV = "plastic_nir_dataset.csv"

CSV_HEADERS = (
    ["Timestamp", "Material_Type", "Sample_Number", "Session_Scan_Number"]
    + [f"NIR_{wl}nm" for wl in NIRSensor.WAVELENGTHS]
    + ["NIR_Temperature"]
)


def get_output_path(custom=None):
    """Return the CSV path, creating parent dirs if needed."""
    if custom:
        path = Path(custom)
    else:
        project_root = Path(__file__).parent.parent
        path = project_root / "logs" / "csvs" / DEFAULT_CSV
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"\nERROR: Cannot create or write to {path.parent}")
        print("  This usually means the folder was created by Docker (owned by root).")
        print(f"  Fix it by running from the project root:")
        print(f"    sudo chown -R $USER:$USER logs/")
        sys.exit(1)
    return path


def count_existing_rows(path):
    """Return number of data rows already in the CSV (0 if file doesn't exist)."""
    if not path.exists():
        return 0
    with open(path, newline='') as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header row


def prompt_material():
    print(f"\nPreset materials: {', '.join(PRESET_MATERIALS)}")
    while True:
        raw = input("Enter material type (or 'q' to finish): ").strip()
        if raw.lower() == 'q':
            return None
        if raw:
            return raw.upper()
        print("  Material name cannot be empty.")


def scan_loop(sensor, material, session_count, writer, csv_file, fixed_material=False, reads=1):
    """
    Interactive scan loop for one material type.

    fixed_material=True means material was passed via --material flag:
      'n' exits instead of switching, since the next material gets its own command.
    reads: number of scans taken per Enter press (default 1).

    Returns (new_session_count, new_sample_num, should_quit).
    """
    print(f"\n[{material}] Place sample under sensor.")
    reads_label = f"{reads} reading{'s' if reads > 1 else ''} per trigger"
    if fixed_material:
        print(f"  Enter = scan ({reads_label}) | 'q' = finish\n")
    else:
        print(f"  Enter = scan ({reads_label}) | 'n' = next material | 'q' = finish\n")

    sample_num = 0

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
            if fixed_material:
                print("  Enter=scan | q=finish")
            else:
                print("  Enter=scan | n=next material | q=finish")
            continue

        # Blank Enter → take `reads` scans in sequence
        for i in range(reads):
            prefix = f"  [{i+1}/{reads}]" if reads > 1 else "  "
            print(f"{prefix} Scanning...", end=" ", flush=True)
            try:
                data = sensor.take_measurement()
            except Exception as e:
                print(f"ERROR: {e}")
                break

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


def main(material_arg=None, output_path=None, reads=1):
    output_path = get_output_path(output_path)
    existing_rows = count_existing_rows(output_path)
    is_new_file = existing_rows == 0

    print("\n" + "=" * 60)
    print("  PLASTIC NIR SCANNER")
    print("=" * 60)
    print(f"  CSV file : {output_path}")
    print(f"  Reads/trigger: {reads}")
    if existing_rows:
        print(f"  Existing : {existing_rows} readings (appending)")
    else:
        print("  Existing : none (new file)")
    print("=" * 60)

    print("\nInitializing NIR sensor...", end=" ", flush=True)
    try:
        sensor = NIRSensor()
    except Exception as e:
        print(f"\nERROR: Could not initialize sensor: {e}")
        sys.exit(1)
    print("ready.\n")

    session_count = 0
    material_counts = {}
    quit_session = False

    # 'a' appends; write header only when creating a new file
    try:
        csv_file_handle = open(output_path, 'a', newline='')
    except PermissionError:
        print(f"\nERROR: Permission denied writing to {output_path}")
        print("  The file or its parent folder is likely owned by root (created by Docker).")
        print("  Fix it by running from the project root:")
        print("    sudo chown -R $USER:$USER logs/")
        sensor.close()
        sys.exit(1)

    with csv_file_handle as csv_file:
        writer = csv.writer(csv_file)
        if is_new_file:
            writer.writerow(CSV_HEADERS)

        try:
            if material_arg:
                # Non-interactive: material provided via --material flag
                material = material_arg.upper()
                session_count, sample_num, _ = scan_loop(
                    sensor, material, session_count, writer, csv_file, fixed_material=True, reads=reads
                )
                material_counts[material] = sample_num
            else:
                # Interactive: prompt for each material
                while not quit_session:
                    material = prompt_material()
                    if material is None:
                        break
                    session_count, sample_num, quit_session = scan_loop(
                        sensor, material, session_count, writer, csv_file, fixed_material=False, reads=reads
                    )
                    material_counts[material] = material_counts.get(material, 0) + sample_num

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
        finally:
            sensor.close()

    total_in_file = existing_rows + session_count
    print("\n" + "=" * 60)
    print("  SESSION COMPLETE")
    print("=" * 60)
    print(f"  Added this run   : {session_count} readings")
    print(f"  Total in file    : {total_in_file} readings")
    print(f"  CSV file         : {output_path}")
    if material_counts:
        print()
        print("  This session:")
        for mat, count in sorted(material_counts.items()):
            label = "reading" if count == 1 else "readings"
            print(f"    {mat:<10} {count} {label}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect NIR spectra for plastic samples — no camera required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python sparkfun/plastic_scanner.py                   # interactive\n"
            "  python sparkfun/plastic_scanner.py -m HDPE           # scan HDPE directly\n"
            "  python sparkfun/plastic_scanner.py -m HDPE -r 3      # 3 scans per Enter\n"
            "  python sparkfun/plastic_scanner.py -m PET -o /data/plastics.csv"
        )
    )
    parser.add_argument(
        "--material", "-m",
        metavar="TYPE",
        help="Material type tag (e.g. HDPE, PET). Skips the interactive prompt."
    )
    parser.add_argument(
        "--reads", "-r",
        metavar="N",
        type=int,
        default=1,
        help="Number of scans taken per Enter press (default: 1)"
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help=f"CSV path (default: logs/csvs/{DEFAULT_CSV})"
    )
    args = parser.parse_args()
    main(material_arg=args.material, output_path=args.output, reads=args.reads)
