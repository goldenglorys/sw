FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.6/dist-packages/"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY docker_requirements.txt .

RUN apt-get update && apt-get install -y \
    # Barcode decoding
    libzbar0 \
    # USB camera support
    v4l-utils \
    # GStreamer (CSI camera for CSI fallback)
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    # I2C tools — needed for SparkFun NIR sensor over Qwiic
    i2c-tools \
    python3-smbus \
    libi2c-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r docker_requirements.txt

COPY . .

# Create log directories
RUN mkdir -p logs/csvs logs/images

CMD ["python3", "camera_app.py"]