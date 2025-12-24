FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ติดตั้ง system dependencies สำหรับ OpenCV และ ImageMagick
RUN apt-get update && apt-get install -y \
    imagemagick \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# อัพเกรด pip
RUN python -m pip install --upgrade pip

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${PORT}

# รัน FastAPI ด้วย Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
