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
- You can supply the material type via the `--material` flag to skip any prompts, or run interactively and type the material name each session.

### Running directly on Jetson (without Docker)

```bash
# Non-interactive — material tag supplied upfront (recommended workflow)
python3 sparkfun/plastic_scanner.py --material HDPE
python3 sparkfun/plastic_scanner.py -m PET

# Interactive — prompts you to type a material name each time
python3 sparkfun/plastic_scanner.py

# Custom CSV path (still appends if the file already exists)
python3 sparkfun/plastic_scanner.py -m HDPE --output /path/to/my_dataset.csv
```

### Running inside Docker

```bash
# Non-interactive (recommended)
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py --material HDPE

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
# Scan HDPE samples — press Enter for each sample, q when done
python3 sparkfun/plastic_scanner.py -m HDPE

# Then scan PET samples — appended to the same CSV
python3 sparkfun/plastic_scanner.py -m PET

# Then LDPE, and so on
python3 sparkfun/plastic_scanner.py -m LDPE
```

### Example session (with `--material` flag)

```
============================================================
  PLASTIC NIR SCANNER
============================================================
  CSV file : logs/csvs/plastic_nir_dataset.csv
  Existing : 6 readings (appending)
============================================================

Initializing NIR sensor... ready.

[HDPE] Place sample under sensor.
  Enter = scan | 'q' = finish

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

### CSV output format

All materials are written to **one persistent CSV file**, with a `Material_Type` column as the label. Filter and group by material in pandas, Excel, or any data tool.

| Column | Description |
|---|---|
| `Timestamp` | Date and time of the scan |
| `Material_Type` | Label you entered (e.g. `HDPE`, `PET`) |
| `Sample_Number` | Per-material counter (resets if you re-enter the same material later in a session) |
| `Session_Scan_Number` | Global counter for the whole session |
| `NIR_410nm` … `NIR_940nm` | 18 calibrated spectral values |
| `NIR_Temperature` | Sensor temperature at scan time |

Output files are saved to `logs/csvs/plastic_nir_<timestamp>.csv` by default.

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