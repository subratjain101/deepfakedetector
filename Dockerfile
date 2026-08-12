FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create DeepFace weights directory
RUN mkdir -p /root/.deepface/weights

# Download FaceNet weights during image build
RUN curl -L --fail --retry 3 --retry-delay 5 \
    -o /root/.deepface/weights/facenet_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5

# Copy application
COPY main.py .
COPY templates ./templates
COPY static ./static

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
