# ใช้ Python 3.11 base image
FROM python:3.11-slim

# ตั้ง environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# อัพเดต package และติดตั้ง dependencies ของ system
RUN apt-get update && apt-get install -y \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# สร้าง working directory
WORKDIR /app

# คัดลอกไฟล์ requirements.txt
COPY requirements.txt .

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดไป container
COPY . .

# เปิด port
EXPOSE ${PORT}

# คำสั่ง run FastAPI ด้วย Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
