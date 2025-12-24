from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import uuid
import os
import subprocess
import shutil

app = FastAPI()

@app.post("/enhance_all_plates")
async def enhance_all_plates(file: UploadFile = File(...)):
    # เตรียมไฟล์ input
    input_path = f"/tmp/{uuid.uuid4()}.jpg"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # โหลดภาพด้วย OpenCV
    img = cv2.imread(input_path)
    if img is None:
        return {"detail": "Cannot read input image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ตรวจสอบ ImageMagick
    if not shutil.which("convert"):
        return {"detail": "ImageMagick 'convert' command not found"}

    # ใช้ Haar Cascade สำหรับป้ายทะเบียน
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    if plate_cascade.empty():
        return {"detail": f"Failed to load cascade: {cascade_path}"}

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,20))

    cropped_files = []

    for i, (x, y, w, h) in enumerate(plates):
        # Crop ป้าย
        crop_img = img[y:y+h, x:x+w]
        crop_path = f"/tmp/crop_{i}_{uuid.uuid4()}.jpg"
        cv2.imwrite(crop_path, crop_img)

        # Enhance ด้วย ImageMagick
        out_path = f"/tmp/enhanced_{i}_{uuid.uuid4()}.jpg"
        cmd = [
            "convert", crop_path,
            "-colorspace", "Gray",
            "-contrast-stretch", "0",
            "-sharpen", "0x1",
            "-threshold", "50%",
            out_path
        ]
        subprocess.run(cmd, check=True)
        cropped_files.append(out_path)

    if not cropped_files:
        os.remove(input_path)
        return {"detail": "No plates detected"}

    # รวมภาพทั้งหมดเป็น 1 ภาพ (stack แนวตั้ง)
    imgs_to_stack = [cv2.imread(f) for f in cropped_files]
    max_width = max(img.shape[1] for img in imgs_to_stack)
    total_height = sum(img.shape[0] for img in imgs_to_stack)

    stacked_img = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    current_y = 0
    for img in imgs_to_stack:
        h, w = img.shape[:2]
        stacked_img[current_y:current_y+h, 0:w] = img
        current_y += h

    final_path = f"/tmp/final_{uuid.uuid4()}.jpg"
    cv2.imwrite(final_path, stacked_img)

    # ลบไฟล์ชั่วคราว
    for f in cropped_files:
        try:
            os.remove(f)
        except:
            pass
    try:
        os.remove(input_path)
    except:
        pass

    return FileResponse(
        final_path,
        media_type="image/jpeg",
        filename="enhanced_all_plates.jpg"
    )
