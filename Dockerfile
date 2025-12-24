# Base image Python slim
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV YOLO_MODEL=yolov8n.pt

# Install system dependencies: OpenCV, ImageMagick, and dependencies for YOLOv8
RUN apt-get update && apt-get install -y \
    imagemagick \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libjpeg-dev \
    libpng-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create tmp directory for temporary files
RUN mkdir -p /tmp

# Expose Railway port
EXPOSE ${PORT}

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
