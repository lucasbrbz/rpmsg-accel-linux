# Build context must be the rpmsg_accel_linux root:
#   docker build -f a53/Dockerfile -t rpmsg-accel-expa .
#
# Run on board:
#   docker run --rm -it \
#     -v /dev:/dev \
#     --device-cgroup-rule "c 199:0 rmw" \
#     rpmsg-accel-expa

FROM torizon/debian-imx8:4

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-numpy \
        python3-psutil \
        python3-tflite-runtime \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/model.tflite     ml/model.tflite
COPY ml/model_labels.npy ml/model_labels.npy
COPY a53/accel_receiver.py .

ENTRYPOINT ["python3", "accel_receiver.py", \
            "--model",  "ml/model.tflite", \
            "--labels", "ml/model_labels.npy"]
