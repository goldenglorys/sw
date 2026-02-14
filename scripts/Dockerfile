FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.6/dist-packages/"

WORKDIR /app

COPY docker_requirements.txt .

RUN apt-get update && apt-get install -y \
    libzbar0 \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r docker_requirements.txt

COPY . .

CMD ["/bin/bash"]
