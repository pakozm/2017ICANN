FROM gcr.io/tensorflow/tensorflow:latest-gpu
RUN pip install keras
# RUN apt-get update && apt-get install nvidia-modprobe && rm -Rf /var/
