FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.6/dist-packages/"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libzbar0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir \
    Pillow==8.4.0 \
    pyzbar==0.1.9 \
    numpy==1.19.5 \
    requests==2.27.1 \
    absl-py

# EXPOSE 8501

#streamlit==1.28.0 \

CMD ["/bin/bash"]
