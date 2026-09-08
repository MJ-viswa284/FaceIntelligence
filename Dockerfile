# Slim Python base -- keeps image size down since TensorFlow/DeepFace
# already add a lot of weight on their own.
FROM python:3.10-slim

WORKDIR /app

# System libs OpenCV needs at runtime (headless server has none of these
# by default -- without them cv2 import fails with "libGL.so.1" errors).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render sets $PORT at runtime -- must bind to it, not a hardcoded 8000.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}