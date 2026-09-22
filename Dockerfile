# Real GPU image for re_polar/: re_polar/core/model_loader.py defaults to
# attn_implementation="flash_attention_2" for any CUDA run, which needs a
# prebuilt wheel (not a normal pip package) -- this Dockerfile exists mainly
# to pin that combo. Base image is the exact devel tag this flash-attn wheel
# is actually validated against elsewhere -- not swapped for the leaner
# -runtime variant, since that combination has never been tested and CUDA/
# cudnn ABI mismatches between runtime/devel variants are a real risk for a
# prebuilt wheel. No compiled kernels beyond flash-attn are needed: re_polar's
# model registry has no Mamba/hybrid-attention models, so (unlike some
# flash-attn-using projects) nothing here needs causal-conv1d or triton-based
# linear-attention kernels -- the devel toolchain just goes unused.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Prebuilt wheel, no compilation -- matches this base image's torch/cuda/
# Python combo (torch 2.11, cuda 12.8, cp312).
RUN pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
