#!/usr/bin/env bash
# Install system dependencies for Pillow & rembg
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7 \
    libwebp-dev \
    libavif-dev \
    libturbojpeg-dev \
    gcc \
    g++ \
    python3-dev

# Upgrade pip & install Python packages
pip install --upgrade pip
pip install -r requirements.txt
