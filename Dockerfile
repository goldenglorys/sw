FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.6/dist-packages/"

WORKDIR /app

COPY docker_requirements.txt .

RUN apt-get update && apt-get install -y \
    libzbar0 \
    python3-opencv \
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

# RUN pip3 install --no-cache-dir -U pip
# RUN pip3 install --no-cache-dir \
#     Pillow==8.4.0 \
#     pyzbar==0.1.9 \
#     numpy==1.19.5 \
#     requests==2.27.1 \
#     absl-py
#     psutil

# EXPOSE 8501

#streamlit==1.28.0 \

CMD ["/bin/bash"]
