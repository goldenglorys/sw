#!/bin/bash
set -e

xhost +local:root > /dev/null 2>&1 || true

# ── Detect which video device to pass in ───
# Finds the first /dev/video* that exists; defaults to /dev/video0
VIDEO_DEVICE="/dev/video0"
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        VIDEO_DEVICE="$dev"
        break
    fi
done
echo "Using camera: $VIDEO_DEVICE"

# ── Build --device flags for ALL video devices (handles video0, video1 etc.) ──
VIDEO_FLAGS=""
for dev in /dev/video*; do
    [ -e "$dev" ] && VIDEO_FLAGS="$VIDEO_FLAGS --device=$dev"
done

# ── I2C device for SparkFun NIR sensor ───────
I2C_FLAGS=""
for dev in /dev/i2c*; do
    [ -e "$dev" ] && I2C_FLAGS="$I2C_FLAGS --device=$dev"
done
if [ -z "$I2C_FLAGS" ]; then
    echo "WARNING: No I2C devices found. NIR sensor will not be available."
else
    echo "I2C devices passed through: $I2C_FLAGS"
fi

docker run -it --rm \
  --gpus all \
  --net=host \
  --privileged \
  $VIDEO_FLAGS \
  $I2C_FLAGS \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -e DISPLAY=$DISPLAY \
  -e PLATFORM_OVERRIDE=jetson \
  -e FORCE_USB=1 \
  -v "$(pwd)":/app \
  barcode-detector \
  python3 camera_app.py