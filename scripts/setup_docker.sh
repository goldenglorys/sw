#!/bin/bash

set -e

echo "================================================="
echo "Jetson Nano Docker Setup for Barcode Detector"
echo "================================================="

# STEP 1: INSTALL DOCKER & NVIDIA CONTAINER RUNTIME
echo -e "\n[INFO] Checking for Docker and NVIDIA Container Runtime..."
if ! command -v docker &> /dev/null || ! dpkg -l | grep -q nvidia-docker2; then
    echo "[INFO] Docker or NVIDIA Container Runtime not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y curl
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Install NVIDIA Container Runtime
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

    docker run --rm --gpus all nvcr.io/nvidia/l4t-base:r32.7.1 nvidia-smi
    
    echo -e "\n\n******************* IMPORTANT *******************"
    echo "You must log out and log back in for the user group changes to take effect."
    echo "After logging back in, re-run this script to continue."
    echo "***************************************************"
    exit 1
else
    echo "[INFO] Docker and NVIDIA Container Runtime are already installed."
fi

# STEP 2: BUILD THE DOCKER IMAGE
echo -e "\n[INFO] Building the 'barcode-detector' Docker image..."
echo "[INFO] This may take several minutes, especially on the first run."

# Ensure the Dockerfile exists in the current directory
if [ ! -f Dockerfile ]; then
    echo "[ERROR] Dockerfile not found in the current directory. Please create it first."
    exit 1
fi

docker build -t barcode-detector .

echo -e "\n================================================="
echo "Setup Complete!"
echo "================================================="
echo -e "\nThe 'barcode-detector' Docker image has been built successfully."
