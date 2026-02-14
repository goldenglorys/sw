#!/bin/bash
# Barcode/QR Detection System - Simple Deployment
# Auto-detects platform and sets up everything

set -e

echo "=================================="
echo "Barcode/QR Detection Setup"
echo "=================================="

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
elif grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    PLATFORM="rpi"
else
    PLATFORM="desktop"
fi

echo "Platform: $PLATFORM"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.7+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if [ "$PLATFORM" = "jetson" ]; then
        python3 -m venv .venv --system-site-packages
    else
        python3 -m venv .venv
    fi
else
    echo "Virtual environment exists"
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Platform-specific system dependencies
if [ "$PLATFORM" = "rpi" ]; then
    echo "Installing Raspberry Pi system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y libzbar0 libatlas-base-dev -qq
elif [ "$PLATFORM" = "jetson" ]; then
    echo "Installing Jetson Nano system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y libzbar0 -qq
else
    # Desktop - try to install libzbar if on Linux
    if [ "$(uname)" = "Linux" ]; then
        echo "Installing system dependencies..."
        sudo apt-get update -qq 2>/dev/null || true
        sudo apt-get install -y libzbar0 -qq 2>/dev/null || true
    fi
fi

# Install Python packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing Python packages..."
    
    if [ "$PLATFORM" = "rpi" ]; then
        # Use headless OpenCV for RPi
        pip install -q opencv-python-headless==4.8.1.78
        pip install -q tensorflow==2.11.0 pillow==10.0.0 pyzbar==0.1.9 numpy==1.23.5 requests==2.31.0 streamlit==1.28.0
    elif [ "$PLATFORM" = "jetson" ]; then
        # Skip TensorFlow on Jetson (should be pre-installed)
        pip install -q pillow==9.5.0 pyzbar==0.1.9 numpy==1.21.6 requests==2.31.0 streamlit==1.28.0
        echo "Note: TensorFlow should be pre-installed on Jetson. If not, install manually."
    else
        # Desktop - install everything from requirements.txt
        pip install -q -r requirements.txt
    fi
else
    echo "WARNING: requirements.txt not found, using default packages..."
    pip install -q tensorflow opencv-python pillow pyzbar numpy requests streamlit
fi

# Verify model weights
echo ""
echo "Checking model weights..."
WEIGHTS_FOUND=0

if [ -f "data/model/yolov3_train_class2_final.weights.h5" ]; then
    echo "✓ Found: data/model/yolov3_train_class2_final.weights.h5"
    WEIGHTS_FOUND=1
elif [ -f "data/model/yolov3_train_class1_final.weights.h5" ]; then
    echo "✓ Found: data/model/yolov3_train_class1_final.weights.h5"
    WEIGHTS_FOUND=1
elif [ -f "data/model/yolov3_train_2class_35.weights.h5" ]; then
    echo "✓ Found: data/model/yolov3_train_2class_35.weights.h5"
    WEIGHTS_FOUND=1
fi

if [ $WEIGHTS_FOUND -eq 0 ]; then
    echo "⚠ WARNING: Model weights not found in data/model/"
    echo "  Expected: yolov3_train_class2_final.weights.h5"
fi

# Test camera
echo ""
echo "Testing camera..."
python3 -c "
import cv2
import sys
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('✓ Camera working (%dx%d)' % (frame.shape[1], frame.shape[0]))
        else:
            print('⚠ Camera opened but cannot read frames')
    else:
        print('⚠ Cannot open camera')
    cap.release()
except Exception as e:
    print('⚠ Camera test failed:', e)
" 2>/dev/null || echo "⚠ Camera test skipped"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To run the application:"
echo ""
echo "  1. Activate environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run camera app:"
echo "     python3 camera_app.py"
echo ""
echo "  3. Run web app:"
echo "     streamlit run app.py"
echo ""
echo "Platform: $PLATFORM"
echo ""