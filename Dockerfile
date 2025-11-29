# Use official Python slim image
FROM python:3.11-slim

# set working dir
WORKDIR /app

# system deps: poppler (pdf2image), tesseract, OpenCV dependencies, imaging libs
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
      # OpenCV dependencies
      libopencv-dev \
      python3-opencv \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching layer
COPY requirements.txt /app/requirements.txt

# install python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Pre-download LayoutLMv3 model (optional, speeds up first inference)
RUN python -c "from transformers import AutoProcessor, AutoModelForTokenClassification; AutoProcessor.from_pretrained('microsoft/layoutlmv3-base'); AutoModelForTokenClassification.from_pretrained('microsoft/layoutlmv3-base')" || echo "Model download failed, will download at runtime"

# copy project files
COPY . /app

# make sure TESSERACT and POPPLER paths are usable in runtime (optional env)
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV POPPLER_PATH=/usr/bin

# Expose port (Render will set $PORT at runtime)
ENV PORT 8000
EXPOSE 8000

# Use a production WSGI server (gunicorn + uvicorn)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api:app", "--bind", "0.0.0.0:${PORT}", "--workers", "1", "--timeout", "120"]
