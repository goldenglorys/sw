# Running the Barcode Detection System with Docker

This guide contains the commands to run the different parts of the application inside the Docker container.

### Prerequisites

1.  You must have a fully set up Jetson Nano with Jetson OS.
2.  You must have successfully run the `./setup_docker.sh` script once.

---

### 1. Running the Live Camera Application (`camera_app.py`)

This command starts the container and gives it access to the Jetson's GPU, camera, and display to show the live video feed.

**Command:**
```bash
docker run -it --rm --gpus all --device /dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY barcode-detector
```

**Inside the container, run:**
```bash
python3 camera_app.py
```

*   **To Stop:** Press `q` in the application window, then type `exit` in the terminal or press `Ctrl+D` to close the container.

---

### 2. Running the Web Application (`app.py`)

This command starts the container, gives it GPU access, and maps the container's port `8501` to your Jetson's port `8501` so you can access it from a browser.

**Command:**
```bash
docker run -it --rm --gpus all -p 8501:8501 barcode-detector
```

**Inside the container, run:**
```bash
streamlit run app.py --server.address=0.0.0.0
```

*   **To Access:** Open a web browser on your Jetson (or another computer on the same network) and navigate to `http://<YOUR_JETSON_IP_ADDRESS>:8501`.
*   **To Stop:** Press `Ctrl+C` in the terminal, then type `exit` or press `Ctrl+D`.

---

### 3. Running the Test Suite (`test_detection.py`)

This is useful for verifying that all components inside the container are working correctly.

**Command:**
```bash
docker run -it --rm --gpus all --device /dev/video0 barcode-detector
```

**Inside the container, run:**
```bash
python3 test_detection.py





```
nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e
```
```

*   This will run all the diagnostic tests and print a summary. The container will exit automatically after the tests are complete if you use `--rm`.
