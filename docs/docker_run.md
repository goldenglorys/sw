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

The script is fully interactive and on-demand:
1. You enter a material type (e.g. `HDPE`)
2. Place the plastic sample under the sensor
3. Press **Enter** to take one scan — takes ~1 second
4. Repeat as many times as you want for that material
5. Type `n` to switch to a new material type
6. Type `q` to finish the session

Every scan is saved immediately to a single CSV file tagged with the material name. Nothing is written unless you explicitly trigger a scan, so there is no risk of capturing empty readings.

### Running directly on Jetson (without Docker)

```bash
# From the project root
python3 sparkfun/plastic_scanner.py

# Custom output path
python3 sparkfun/plastic_scanner.py --output /path/to/my_dataset.csv
```

### Running inside Docker

```bash
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py
```

> No `--gpus`, no `--device=/dev/video*`, no display flags needed — this script only uses the I2C sensor.  
> Change `/dev/i2c-1` to `/dev/i2c-0` if that's what `ls /dev/i2c*` shows on your Jetson.

### Custom output path in Docker

```bash
docker run -it --rm \
  --privileged \
  --device=/dev/i2c-1 \
  -v $(pwd):/app \
  barcode-detector \
  python3 sparkfun/plastic_scanner.py --output /app/logs/csvs/my_plastics.csv
```

### Example session

```
============================================================
  PLASTIC NIR SCANNER
============================================================
  Output: logs/csvs/plastic_nir_20260515_143022.csv
============================================================

Initializing NIR sensor... ready.

Preset materials: HDPE, LDPE, PET, PP, PS, PVC
Enter material type (or 'q' to finish session): HDPE

[HDPE] Place sample under sensor.
  Enter = scan | 'n' = next material | 'q' = finish session

  [HDPE] >
  Scanning... done.  peak=730nm (88.4), temp=28.1°C  [reading #1 saved]
  [HDPE] >
  Scanning... done.  peak=730nm (87.9), temp=28.2°C  [reading #2 saved]
  [HDPE] > n

Preset materials: HDPE, LDPE, PET, PP, PS, PVC
Enter material type (or 'q' to finish session): PET

[PET] Place sample under sensor.
  [PET] >
  Scanning... done.  peak=810nm (102.3), temp=28.3°C  [reading #3 saved]
  [PET] > q

============================================================
  SESSION COMPLETE
============================================================
  Total readings : 3
  Output file    : logs/csvs/plastic_nir_20260515_143022.csv

  Breakdown:
    HDPE       2 readings
    PET        1 reading
============================================================
```

### CSV output format

All materials are written to **one CSV file** per session, with a `Material_Type` column as the label. You can filter and group by material in pandas, Excel, or any data tool.

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