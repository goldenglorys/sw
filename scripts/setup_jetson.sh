#!/bin/bash

# ========================================
# Jetson Nano Deployment Helper Script
# ========================================

set -e  # Exit on error

echo "🚀 Barcode Detection - Jetson Nano Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    print_error "This script is designed for Jetson Nano!"
    print_info "Detected platform: $(uname -s)"
    exit 1
fi

print_success "Jetson Nano detected!"
echo ""

# Step 1: Update system
# print_info "Step 1/6: Updating system packages..."
# sudo apt update
# print_success "System updated"
# echo ""

# Step 2: Install system dependencies
# print_info "Step 2/6: Installing system dependencies..."
# sudo apt install -y \
#     python3-pip \
#     python3-dev \
#     python3-opencv \
#     libzbar0 \
#     libzbar-dev \
#     v4l-utils \
#     gstreamer1.0-tools \
#     gstreamer1.0-plugins-base \
#     gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-ugly \
#     gstreamer1.0-libav \
#     libgstreamer1.0-dev

# print_success "System dependencies installed"
# echo ""

# Step 3: Create virtual environment
# print_info "Step 3/6: Setting up Python virtual environment..."
# if [ ! -d ".venv" ]; then
#     python3 -m venv .venv
#     print_success "Virtual environment created"
# else
#     print_info "Virtual environment already exists"
# fi

# source .venv/bin/activate
# print_success "Virtual environment activated"
# echo ""

# Step 4: Upgrade pip
# print_info "Step 4/6: Upgrading pip..."
# pip3 install --upgrade pip
# print_success "Pip upgraded"
# echo ""

# Step 5: Install Python dependencies
# print_info "Step 5/6: Installing Python dependencies..."
# pip3 install scikit-build
# pip3 install opencv-python
# pip3 install pillow
# pip3 install pyzbar
# pip3 install requests
# pip3 install numpy

# print_success "Python dependencies installed"
# echo ""

# Step 6: Install TensorFlow for Jetson
print_info "Step 6/6: Installing TensorFlow for Jetson..."
print_info "This may take a few minutes..."

# Check JetPack version
JETPACK_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-jetpack 2>/dev/null || echo "unknown")
print_info "JetPack version: $JETPACK_VERSION"

# Install TensorFlow
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow

print_success "TensorFlow installed"
echo ""

# Verify camera
print_info "Verifying camera connection..."
if ls /dev/video* 1> /dev/null 2>&1; then
    print_success "Camera device found: $(ls /dev/video*)"
else
    print_error "No camera device found!"
    print_info "Please connect CSI or USB camera"
fi
echo ""

# Check model files
print_info "Checking model files..."
if [ -d "checkpoints" ] && [ -d "model" ]; then
    print_success "Model directories found"
    if ls checkpoints/*.h5 1> /dev/null 2>&1; then
        print_success "Model weights found: $(ls checkpoints/*.h5 | wc -l) files"
    else
        print_error "No model weights (.h5) found in checkpoints/"
        print_info "Make sure to copy your trained model from Mac!"
    fi
else
    print_error "Model directories not found"
    print_info "Make sure model/ and checkpoints/ directories exist"
fi
echo ""

# Enable max performance (optional)
read -p "Enable maximum performance mode? (10W, recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Setting Jetson to max performance..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
    print_success "Max performance mode enabled"
else
    print_info "Skipping performance mode"
fi
echo ""

# Final summary
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
print_success "Your Jetson Nano is ready for barcode detection!"
echo ""
echo "To run the application:"
echo "  1. source .venv/bin/activate"
echo "  2. python3 camera_app.py"
echo ""
print_info "Press 'q' or ESC to quit the application"
print_info "Logs will be saved to barcode_log_*.csv"
echo ""

# Test camera (optional)
read -p "Test camera now? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Testing CSI camera with GStreamer..."
    print_info "You should see camera output. Press Ctrl+C to stop."
    sleep 2
    
    gst-launch-1.0 nvarguscamerasrc ! \
        'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
        nvvidconv ! 'video/x-raw,format=BGRx' ! \
        videoconvert ! autovideosink || {
        print_error "CSI camera test failed"
        print_info "Trying USB camera..."
        gst-launch-1.0 v4l2src device=/dev/video0 ! \
            video/x-raw,width=640,height=480 ! \
            videoconvert ! autovideosink || {
            print_error "USB camera test also failed"
            print_info "Please check camera connection"
        }
    }
fi

echo ""
print_success "All done! Happy scanning! 📦🔍"
