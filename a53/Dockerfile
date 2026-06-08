# Build context must be the rpmsg_accel_linux root:
#   docker build -f a53/Dockerfile -t rpmsg-accel-expa .
#
# Run on board:
#   docker run --rm -it \
#     -v /dev:/dev \
#     --device-cgroup-rule "c 199:0 rmw" \
#     -v ~/exp-a-results:/app \
#     rpmsg-accel-expa

ARG IMAGE_ARCH=linux/arm64

FROM --platform=$IMAGE_ARCH torizon/debian-imx8:4

ARG WORKDIR_PATH="/app"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1      \
    PYTHONUNBUFFERED=1

WORKDIR ${WORKDIR_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3                  \
        python3-venv             \
        python3-pip              \
        python3-numpy            \
        libtensorflow-lite2.16.2 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${WORKDIR_PATH}/.venv --system-site-packages

RUN . ${WORKDIR_PATH}/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tflite-runtime psutil

COPY ml/model.tflite     ml/model.tflite
COPY ml/model_labels.npy ml/model_labels.npy
COPY a53/accel_receiver.py .

ENTRYPOINT ["/app/.venv/bin/python3", "accel_receiver.py", \
            "--model",  "ml/model.tflite", \
            "--labels", "ml/model_labels.npy"]
