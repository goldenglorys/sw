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

## 4. Testing Individual Components

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

## 5. Quick Reference

| Situation | What to do |
|---|---|
| Changed a `.py` file | Run `docker run` again — no rebuild |
| Changed `Dockerfile` or requirements | `docker build -t barcode-detector .` |
| NIR sensor not found | Check `ls /dev/i2c*` and match `--device` flag |
| Wrong camera / black screen | Check `ls /dev/video*`, try `video0` then `video1` |
| Display not working | Run `xhost +local:root` before `docker run` |
| Run headless (no display) | Remove `-e DISPLAY` and `-v /tmp/.X11-unix` flags |