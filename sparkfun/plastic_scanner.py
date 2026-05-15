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

    # Add a label (e.g. condition, sample description)
    python sparkfun/plastic_scanner.py -m HDPE --label clean_bottle
    python sparkfun/plastic_scanner.py -m HDPE -l dirty_cap

    # Take 3 readings per Enter press
    python sparkfun/plastic_scanner.py -m HDPE -l clean -r 3

    # Custom CSV path (still appends if file exists)
    python sparkfun/plastic_scanner.py -m HDPE --output /path/to/dataset.csv
"""
import sys
import os
import csv
import select
import time
import argparse
from datetime import datetime
from pathlib import Path

# Works whether run from project root or from sparkfun/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nir_sensor import NIRSensor


PRESET_MATERIALS = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
DEFAULT_CSV = "plastic_nir_dataset.csv"
AUTO_QUIT_SECONDS = 3  # countdown after reads complete in fixed-material mode

CSV_HEADERS = (
    ["Timestamp", "Material_Type", "Label", "Sample_Number", "Session_Scan_Number"]
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
        print("  Fix it by running from the project root:")
        print("    sudo chown -R $USER:$USER logs/")
        sys.exit(1)
    return path


def count_existing_rows(path):
    """Return number of data rows already in the CSV (0 if file doesn't exist)."""
    if not path.exists():
        return 0
    with open(path, newline='') as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header row


def warmup_sensor(sensor):
    """
    Take one silent measurement and discard the result.
    The AS7265x runs in CONTINUOUS mode and may be mid-cycle at init time;
    this first call syncs the state so LEDs fire correctly on real scans.
    """
    print("Warming up sensor...", end=" ", flush=True)
    try:
        sensor.take_measurement()
    except Exception:
        pass  # warmup failure is non-fatal
    print("done.\n")


def prompt_material():
    print(f"\nPreset materials: {', '.join(PRESET_MATERIALS)}")
    while True:
        raw = input("Enter material type (or 'q' to finish): ").strip()
        if raw.lower() == 'q':
            return None
        if raw:
            return raw.upper()
        print("  Material name cannot be empty.")


def prompt_label():
    raw = input("Enter label (e.g. clean_bottle, dirty_cap) or press Enter to skip: ").strip()
    return raw if raw else ""


def input_with_timeout(prompt, timeout):
    """
    Show prompt and wait up to `timeout` seconds for a line of input.
    Returns the stripped line, or None if time runs out.
    Works on Linux/Jetson via select(); falls back to plain input() elsewhere.
    """
    print(prompt, end="", flush=True)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip().lower()
        print()  # newline after countdown expires
        return None
    except (AttributeError, OSError):
        # select() not available (Windows) — fall back to blocking input
        return input().strip().lower()


def take_batch(sensor, material, label, sample_num, session_count, reads, writer, csv_file):
    """
    Fire `reads` sequential scans for `material`.
    Returns (updated_sample_num, updated_session_count, had_error).
    """
    had_error = False
    for i in range(reads):
        prefix = f"  [{i+1}/{reads}]" if reads > 1 else "  "
        print(f"{prefix} Scanning...", end=" ", flush=True)
        try:
            data = sensor.take_measurement()
        except Exception as e:
            print(f"ERROR: {e}")
            had_error = True
            break

        sample_num += 1
        session_count += 1

        iso_ts = datetime.fromtimestamp(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = (
            [iso_ts, material, label, sample_num, session_count]
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

    return sample_num, session_count, had_error


def scan_loop_fixed(sensor, material, label, session_count, writer, csv_file, reads):
    """
    Scan loop when material is provided via --material flag.
    After each batch completes, auto-quits after AUTO_QUIT_SECONDS unless
    the user presses Enter to trigger another batch.
    Returns (new_session_count, new_sample_num).
    """
    reads_label = f"{reads} reading{'s' if reads > 1 else ''} per trigger"
    label_display = f", label={label}" if label else ""
    print(f"\n[{material}{label_display}] Place sample under sensor.")
    print(f"  Enter = scan ({reads_label}) | 'q' = quit\n")

    sample_num = 0
    first_trigger = True

    while True:
        if first_trigger:
            # Wait indefinitely for the first scan
            try:
                cmd = input(f"  [{material}] > ").strip().lower()
            except EOFError:
                break
            first_trigger = False
        else:
            # After a completed batch: auto-quit countdown
            cmd = input_with_timeout(
                f"\n  [{material}] > Press Enter to scan again, "
                f"or auto-quitting in {AUTO_QUIT_SECONDS}s... ",
                AUTO_QUIT_SECONDS
            )
            if cmd is None:
                # Timed out — auto-quit
                break

        if cmd == 'q':
            break
        if cmd != '':
            print("  Enter=scan | q=quit")
            first_trigger = first_trigger  # keep waiting
            continue

        sample_num, session_count, _ = take_batch(
            sensor, material, label, sample_num, session_count, reads, writer, csv_file
        )

    return session_count, sample_num


def scan_loop_interactive(sensor, material, label, session_count, writer, csv_file, reads):
    """
    Scan loop for interactive mode (no --material flag).
    Keeps scanning until 'n' (next material) or 'q' (quit).
    Returns (new_session_count, new_sample_num, should_quit).
    """
    reads_label = f"{reads} reading{'s' if reads > 1 else ''} per trigger"
    label_display = f", label={label}" if label else ""
    print(f"\n[{material}{label_display}] Place sample under sensor.")
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
            print("  Enter=scan | n=next material | q=finish")
            continue

        sample_num, session_count, _ = take_batch(
            sensor, material, label, sample_num, session_count, reads, writer, csv_file
        )


def main(material_arg=None, label_arg=None, output_path=None, reads=1):
    output_path = get_output_path(output_path)
    existing_rows = count_existing_rows(output_path)
    is_new_file = existing_rows == 0

    print("\n" + "=" * 60)
    print("  PLASTIC NIR SCANNER")
    print("=" * 60)
    print(f"  CSV file     : {output_path}")
    print(f"  Reads/trigger: {reads}")
    if label_arg:
        print(f"  Label        : {label_arg}")
    if existing_rows:
        print(f"  Existing     : {existing_rows} readings (appending)")
    else:
        print("  Existing     : none (new file)")
    print("=" * 60)

    print("\nInitializing NIR sensor...", end=" ", flush=True)
    try:
        sensor = NIRSensor()
    except Exception as e:
        print(f"\nERROR: Could not initialize sensor: {e}")
        sys.exit(1)
    print("ready.")

    # Warmup scan — syncs continuous-mode sensor state so LEDs fire correctly
    warmup_sensor(sensor)

    session_count = 0
    material_counts = {}
    quit_session = False

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
                material = material_arg.upper()
                label = label_arg if label_arg is not None else prompt_label()
                session_count, sample_num = scan_loop_fixed(
                    sensor, material, label, session_count, writer, csv_file, reads
                )
                material_counts[material] = sample_num
            else:
                while not quit_session:
                    material = prompt_material()
                    if material is None:
                        break
                    label = label_arg if label_arg is not None else prompt_label()
                    session_count, sample_num, quit_session = scan_loop_interactive(
                        sensor, material, label, session_count, writer, csv_file, reads
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
            "  python sparkfun/plastic_scanner.py                        # interactive\n"
            "  python sparkfun/plastic_scanner.py -m HDPE                # scan HDPE, auto-quit\n"
            "  python sparkfun/plastic_scanner.py -m HDPE -l clean_cap   # with label\n"
            "  python sparkfun/plastic_scanner.py -m HDPE -l dirty -r 3  # label + 3 reads\n"
            "  python sparkfun/plastic_scanner.py -m PET -o /data/plastics.csv"
        )
    )
    parser.add_argument(
        "--material", "-m",
        metavar="TYPE",
        help="Material type tag (e.g. HDPE, PET). Skips the interactive prompt."
    )
    parser.add_argument(
        "--label", "-l",
        metavar="LABEL",
        default=None,
        help="Free-text label for this scan (e.g. clean_bottle, dirty_cap). "
             "Prompted interactively if not supplied."
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
    main(material_arg=args.material, label_arg=args.label, output_path=args.output, reads=args.reads)
