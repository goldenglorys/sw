#!/usr/bin/env python3
"""
Platform detection and camera source configuration.
Auto-detects USB vs CSI camera — works out of the box.
"""
import os
import cv2


def detect_platform():
    """
    Automatically detect the platform.
    Can be overridden with PLATFORM_OVERRIDE env var.

    Returns:
        str: 'desktop', 'jetson', or 'rpi'
    """
    override = os.environ.get("PLATFORM_OVERRIDE", "").lower()
    if override in ("jetson", "rpi", "desktop"):
        print(f"Platform override: {override}")
        return override

    if os.path.exists("/etc/nv_tegra_release"):
        return "jetson"

    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                if "raspberry pi" in f.read().lower():
                    return "rpi"
        except:
            pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            if "tegra" in f.read().lower():
                return "jetson"
    except:
        pass

    return "desktop"


def _find_usb_camera():
    """
    Scan /dev/video* and return the first index that actually opens
    and returns a frame. Returns 0 as fallback even if nothing found.
    """
    import glob

    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* devices found, defaulting to index 0")
        return 0

    for dev in devices:
        # Extract index number from e.g. /dev/video0
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue

        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Auto-detected USB camera at /dev/video{idx}")
                return idx

    print("Could not verify any camera, defaulting to index 0")
    return 0


def _get_csi_pipeline(width=1280, height=720, fps=30):
    """GStreamer pipeline for Jetson CSI camera."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def get_camera_source(platform_type=None):
    """
    Get the appropriate camera source for the platform.

    Logic:
      - If CAMERA_SOURCE env var is set, use it directly (manual override).
      - If FORCE_USB env var is set to '1', always use USB auto-detect.
      - On Jetson: try CSI first, fall back to USB automatically.
      - Everywhere else: USB auto-detect.

    Returns:
        int or str: camera index (int) or GStreamer pipeline string (str)
    """
    if platform_type is None:
        platform_type = detect_platform()

    # Manual override — useful for testing
    manual = os.environ.get("CAMERA_SOURCE", "").strip()
    if manual:
        print(f"CAMERA_SOURCE override: {manual}")
        try:
            return int(manual)  # numeric index
        except ValueError:
            return manual  # GStreamer string

    force_usb = os.environ.get("FORCE_USB", "0") == "1"

    if platform_type == "jetson" and not force_usb:
        # Try CSI pipeline first
        pipeline = _get_csi_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("CSI camera detected and working.")
                return pipeline
        cap.release()
        print("CSI camera not available, falling back to USB auto-detect...")

    # USB auto-detect (works on Jetson USB, desktop, RPi)
    return _find_usb_camera()


def get_opencv_backend(platform_type=None):
    """Get the appropriate OpenCV backend for VideoCapture."""
    if platform_type is None:
        platform_type = detect_platform()

    if platform_type == "jetson":
        source = get_camera_source(platform_type)
        if isinstance(source, str):  # GStreamer pipeline
            return cv2.CAP_GSTREAMER
    return cv2.CAP_V4L2  # works for USB on Linux; OpenCV falls back gracefully


# ── Quick camera test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Camera Auto-Detection Test")
    print("=" * 50)

    platform = detect_platform()
    print(f"Platform : {platform}")

    source = get_camera_source(platform)
    print(f"Source   : {source}")

    backend = get_opencv_backend(platform)
    cap = (
        cv2.VideoCapture(source, backend)
        if isinstance(source, str)
        else cv2.VideoCapture(source)
    )

    if not cap.isOpened():
        print("FAIL: Could not open camera.")
    else:
        ret, frame = cap.read()
        if ret:
            print(f"OK  : {frame.shape[1]}x{frame.shape[0]} frame captured")
            cv2.imwrite("/tmp/camera_test.jpg", frame)
            print("     Saved to /tmp/camera_test.jpg")
        else:
            print("FAIL: Camera opened but no frame returned.")
        cap.release()
