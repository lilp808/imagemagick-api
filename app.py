from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import uuid
import os
import subprocess
import shutil
import traceback

app = FastAPI()

@app.post("/enhance_all_plates")
async def enhance_all_plates(file: UploadFile = File(...)):
    try:
        # Prepare input
        input_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Check ImageMagick
        if not shutil.which("convert"):
            return {"detail": "ImageMagick 'convert' command not found"}

        # Load image
        img = cv2.imread(input_path)
        if img is None:
            return {"detail": "Cannot read input image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        plate_cascade = cv2.CascadeClassifier(cascade_path)
        if plate_cascade.empty():
            return {"detail": f"Haar cascade not loaded: {cascade_path}"}

        # Detect plates
        plates = plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,20)
        )

        if len(plates) == 0:
            os.remove(input_path)
            return {"detail": "No plates detected"}

        cropped_files = []

        # Crop each plate
        for i, (x, y, w, h) in enumerate(plates):
            crop_img = img[y:y+h, x:x+w]
            crop_path = f"/tmp/crop_{i}_{uuid.uuid4()}.jpg"
            cv2.imwrite(crop_path, crop_img)
            cropped_files.append(crop_path)

        # Stack cropped images vertically
        imgs_to_stack = [cv2.imread(f) for f in cropped_files]
        max_width = max(img.shape[1] for img in imgs_to_stack)
        total_height = sum(img.shape[0] for img in imgs_to_stack)

        stacked_img = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        current_y = 0
        for img_crop in imgs_to_stack:
            h, w = img_crop.shape[:2]
            stacked_img[current_y:current_y+h, 0:w] = img_crop
            current_y += h

        stacked_path = f"/tmp/stacked_{uuid.uuid4()}.jpg"
        cv2.imwrite(stacked_path, stacked_img)

        # Enhance with ImageMagick
        final_path = f"/tmp/final_{uuid.uuid4()}.jpg"
        cmd = [
            "convert", stacked_path,
            "-colorspace", "Gray",
            "-contrast-stretch", "0",
            "-sharpen", "0x1",
            final_path
        ]
        subprocess.run(cmd, check=True)

        # Clean temp files
        for f in cropped_files + [input_path, stacked_path]:
            try:
                os.remove(f)
            except:
                pass

        return FileResponse(
            final_path,
            media_type="image/jpeg",
            filename="enhanced_all_plates.jpg"
        )

    except Exception as e:
        tb = traceback.format_exc()
        return {"detail": str(e), "traceback": tb}
