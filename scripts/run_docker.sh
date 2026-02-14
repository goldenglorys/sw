#!/bin/bash
set -e

xhost +local:root > /dev/null

docker run -it --rm \
  --gpus all \
  --net=host \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -e DISPLAY=$DISPLAY \
  -e PLATFORM_OVERRIDE=jetson \
  -e GST_DEBUG=1 \
  -v "$(pwd)":/app \
  barcode-detector
