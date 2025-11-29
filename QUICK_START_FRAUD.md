# Quick Start - Fraud Detection

## Installation

### Dockerfile (apt-get commands)

```dockerfile
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      poppler-utils \
      tesseract-ocr \
      libtiff5-dev \
      libjpeg-dev \
      zlib1g-dev \
      libpng-dev \
      libwebp-dev \
      pkg-config \
      git \
      libopencv-dev \
      python3-opencv \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

### pip install (already in requirements.txt)

```bash
pip install opencv-python-headless pdf2image
```

## Test train_sample_13

```bash
# CLI test
python test_fraud.py data/raw/training_samples/train_sample_13.pdf

# Unit test
python -m unittest tests.test_fraud_detection.TestFraudDetection.test_train_sample_13_flagged -v
```

## Output Location

Debug images saved to: `local_test_outputs/debug_masks/`


