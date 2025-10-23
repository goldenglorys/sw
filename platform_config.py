import os


def detect_platform():
    """
    Automatically detect the platform.
    
    Returns:
        str: 'desktop', 'jetson', or 'rpi'
    """
    # Check for Jetson Nano
    if os.path.exists('/etc/nv_tegra_release'):
        return 'jetson'
    
    # Check for Raspberry Pi
    if os.path.exists('/proc/device-tree/model'):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                if 'raspberry pi' in f.read().lower():
                    return 'rpi'
        except:
            pass
    
    # Check for Jetson via tegra in cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'tegra' in f.read().lower():
                return 'jetson'
    except:
        pass
    
    # Default to desktop
    return 'desktop'


def get_camera_source(platform_type=None):
    """
    Get the appropriate camera source for the platform.
    
    Returns:
        Camera source (int or string)
    """
    if platform_type is None:
        platform_type = detect_platform()
    
    if platform_type == 'jetson':
        # CSI camera with GStreamer pipeline
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    else:
        # Standard USB camera (works for desktop and RPi)
        return 0


def get_opencv_backend(platform_type=None):
    """Get the appropriate OpenCV backend."""
    if platform_type is None:
        platform_type = detect_platform()
    
    if platform_type == 'jetson':
        try:
            import cv2
            return cv2.CAP_GSTREAMER
        except:
            return None
    return None