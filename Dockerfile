FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir \
    Pillow==8.4.0 \
    pyzbar==0.1.9 \
    streamlit==1.28.0 \
    numpy==1.21.6 \
    requests==2.31.0 \
    absl-py

# EXPOSE 8501

CMD ["/bin/bash"]
