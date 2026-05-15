# Docker Run Guide
**Barcode Detection + NIR Sensor — Jetson Nano**

---

## 1. One-Time Setup

Check your device indices first:
```bash
ls /dev/i2c*      # usually /dev/i2c-1
ls /dev/video*    # usually /dev/video0
```

Build the image (from project root where `Dockerfile` lives):
```bash
docker build -t barcode-detector .
```

> Takes several minutes the first time.

---

## 2. Subsequent Runs

**Changed a `.py` file?** → Just run the app again, no rebuild needed. The project folder is mounted live.

**Changed `Dockerfile` or `docker_requirements.txt`?** → Rebuild:
```bash
docker build -t barcode-detector .
```

---

## 3. Running the Application

### Camera App (main)
```bash
docker run -it --rm \
  --gpus all \
  --net=host \
  --privileged \
  --device=/dev/video0 \
  --device=/dev/i2c-1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -e DISPLAY=$DISPLAY \
  -e PLATFORM_OVERRIDE=jetson \
  -e FORCE_USB=1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 camera_app.py
```
> Press `q` in the app window to stop.  
> Change `/dev/i2c-1` to `/dev/i2c-0` if that's what your Jetson shows.

### Shortcut — use `run_docker.sh`
```bash
chmod +x run_docker.sh
./run_docker.sh
```
Auto-detects all video and I2C devices and runs the camera app.

### Web App (Streamlit)
```bash
docker run -it --rm \
  --gpus all \
  --net=host \
  --privileged \
  --device=/dev/video0 \
  --device=/dev/i2c-1 \
  -e PLATFORM_OVERRIDE=jetson \
  -e FORCE_USB=1 \
  -v $(pwd):/app \
  barcode-detector \
  streamlit run app.py --server.address=0.0.0.0
```
> Access at `http://<JETSON_IP>:8501` from any browser on the same network.

---

## 4. Plastic NIR Data Collection (No Camera)

Use this when you want to scan different plastic materials and build a labelled dataset — no camera or barcode detection involved.

### How it works

- All runs append to **one persistent CSV file** (`logs/csvs/plastic_nir_dataset.csv`). Every time you run the script, new readings are added to the same file — nothing is overwritten.
- Scans are **on-demand**: press **Enter** when the sample is ready. Nothing is saved unless you trigger it, so empty/air readings are impossible.
- You can supply the material type via `--material` and an optional free-text label via `--label` to skip all prompts, or run interactively and type both each session.

### Running directly on Jetson (without Docker)

```bash
# Material only
python3 sparkfun/plastic_scanner.py -m HDPE

# Material + label
python3 sparkfun/plastic_scanner.py -m HDPE --label clean_bottle
python3 sparkfun/plastic_scanner.py -m HDPE -l dirty_cap

# Material + label + multiple reads per trigger
python3 sparkfun/plastic_scanner.py -m HDPE -l clean -r 3

# All flags together
python3 sparkfun/plastic_scanner.py -m HDPE -l clean_bottle -r 3 --output /path/to/my_dataset.csv

# Interactive — prompts for material and label each session
python3 sparkfun/plastic_scanner.py
```

### Running inside Docker

```bash
# Material only
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py -m HDPE

# Material + label
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py -m HDPE -l clean_bottle

# Material + label + 3 reads per trigger
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py -m HDPE -l clean -r 3

# Interactive
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py
```

> No `--gpus`, no `--device=/dev/video*`, no display flags needed — this script only uses the I2C sensor.  
> Change `/dev/i2c-1` to `/dev/i2c-0` if that's what `ls /dev/i2c*` shows on your Jetson.

### Typical workflow for collecting a full dataset

```bash
# Scan clean HDPE bottles — 3 reads per placement
python3 sparkfun/plastic_scanner.py -m HDPE -l clean_bottle -r 3

# Scan dirty HDPE caps (same material, different label)
python3 sparkfun/plastic_scanner.py -m HDPE -l dirty_cap -r 3

# Then PET samples
python3 sparkfun/plastic_scanner.py -m PET -l clean -r 3

# Then LDPE, and so on
python3 sparkfun/plastic_scanner.py -m LDPE -l clean -r 3
```

**When to use `--label`:** Use it to distinguish samples of the same material that differ in condition, shape, or origin — e.g. `clean_bottle` vs `dirty_cap` both being HDPE. The label is a free-text field so you can write anything. If you don't need it, just omit the flag or press Enter to skip the prompt.

**When to use `--reads`:** If you want multiple spectral readings of the same physical placement (e.g. to average or check variance later), set `-r 3` or higher. Each reading is a separate row in the CSV. Default is 1.

### Example session — single read per trigger (`-m HDPE`)

```
============================================================
  PLASTIC NIR SCANNER
============================================================
  CSV file     : logs/csvs/plastic_nir_dataset.csv
  Reads/trigger: 1
  Existing     : 6 readings (appending)
============================================================

Initializing NIR sensor... ready.

[HDPE] Place sample under sensor.
  Enter = scan (1 reading per trigger) | 'q' = finish

  [HDPE] >
   Scanning... done.  peak=730nm (88.4), temp=28.1°C  [reading #1 saved]
  [HDPE] >
   Scanning... done.  peak=730nm (87.9), temp=28.2°C  [reading #2 saved]
  [HDPE] > q

============================================================
  SESSION COMPLETE
============================================================
  Added this run   : 2 readings
  Total in file    : 8 readings
  CSV file         : logs/csvs/plastic_nir_dataset.csv

  This session:
    HDPE       2 readings
============================================================
```

### Example session — 3 reads per trigger (`-m HDPE -r 3`)

```
============================================================
  PLASTIC NIR SCANNER
============================================================
  CSV file     : logs/csvs/plastic_nir_dataset.csv
  Reads/trigger: 3
  Existing     : 8 readings (appending)
============================================================

Initializing NIR sensor... ready.

[HDPE] Place sample under sensor.
  Enter = scan (3 readings per trigger) | 'q' = finish

  [HDPE] >
  [1/3] Scanning... done.  peak=730nm (88.4), temp=28.1°C  [reading #1 saved]
  [2/3] Scanning... done.  peak=730nm (87.6), temp=28.1°C  [reading #2 saved]
  [3/3] Scanning... done.  peak=730nm (88.1), temp=28.2°C  [reading #3 saved]
  [HDPE] > q

============================================================
  SESSION COMPLETE
============================================================
  Added this run   : 3 readings
  Total in file    : 11 readings
  CSV file         : logs/csvs/plastic_nir_dataset.csv

  This session:
    HDPE       3 readings
============================================================
```

### CSV output format

All materials are written to **one persistent CSV file**. Filter and group by `Material_Type` or `Label` in pandas, Excel, or any data tool.

| Column | Description |
|---|---|
| `Timestamp` | Date and time of the scan |
| `Material_Type` | Material tag (e.g. `HDPE`, `PET`) — from `--material` or interactive prompt |
| `Label` | Free-text label (e.g. `clean_bottle`, `dirty_cap`) — from `--label` or interactive prompt, blank if skipped |
| `Sample_Number` | Per-material counter for this session |
| `Session_Scan_Number` | Global counter for the whole run |
| `NIR_410nm` … `NIR_940nm` | 18 calibrated spectral values |
| `NIR_Temperature` | Sensor temperature at scan time |

Output is saved to `logs/csvs/plastic_nir_dataset.csv` by default (appended across all runs).

### Supported material names

The script shows a preset list (`HDPE`, `LDPE`, `PET`, `PP`, `PS`, `PVC`) but you can type **any name** freely — it is treated as a plain text tag. You can also re-enter a material name later in the same session to continue adding readings to it.

---

## 5. Testing Individual Components

### Camera auto-detect test
```bash
docker run -it --rm --gpus all --privileged \
  --device=/dev/video0 -e FORCE_USB=1 \
  -v $(pwd):/app barcode-detector \
  python3 platform_config.py
```

### NIR sensor test
```bash
docker run -it --rm --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app barcode-detector \
  python3 sparkfun/diagnostics.py
```

### GStreamer test (CSI cameras only — not USB)
```bash
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! 'video/x-raw,format=BGRx' ! \
  videoconvert ! autovideosink
```

---

## 6. Quick Reference

| Situation | What to do |
|---|---|
| Changed a `.py` file | Run `docker run` again — no rebuild |
| Changed `Dockerfile` or requirements | `docker build -t barcode-detector .` |
| NIR sensor not found | Check `ls /dev/i2c*` and match `--device` flag |
| Wrong camera / black screen | Check `ls /dev/video*`, try `video0` then `video1` |
| Display not working | Run `xhost +local:root` before `docker run` |
| Run headless (no display) | Remove `-e DISPLAY` and `-v /tmp/.X11-unix` flags |