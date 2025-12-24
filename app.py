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

@app.post("/enhance_all_plate_like")
async def enhance_all_plate_like(file: UploadFile = File(...)):
    try:
        # 1️⃣ Prepare input
        input_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 2️⃣ Check ImageMagick
        if not shutil.which("convert"):
            return {"detail": "ImageMagick 'convert' command not found"}

        # 3️⃣ Load image
        img = cv2.imread(input_path)
        if img is None:
            return {"detail": "Cannot read input image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        # 4️⃣ Threshold + find contours
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cropped_files = []

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            # filter approximate plate shapes
            if 2 < ratio < 6 and w > 30 and h > 10:
                crop_img = img[y:y+h, x:x+w]
                crop_path = f"/tmp/crop_{i}_{uuid.uuid4()}.jpg"
                cv2.imwrite(crop_path, crop_img)
                cropped_files.append(crop_path)

        if len(cropped_files) == 0:
            os.remove(input_path)
            return {"detail": "No plate-like objects detected"}

        # 5️⃣ Stack cropped images vertically
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

        # 6️⃣ Enhance with ImageMagick
        final_path = f"/tmp/final_{uuid.uuid4()}.jpg"
        cmd = [
            "convert", stacked_path,
            "-colorspace", "Gray",
            "-contrast-stretch", "0",
            "-sharpen", "0x1",
            final_path
        ]
        subprocess.run(cmd, check=True)

        # 7️⃣ Clean temp files
        for f in cropped_files + [input_path, stacked_path]:
            try:
                os.remove(f)
            except:
                pass

        # 8️⃣ Return final image
        return FileResponse(
            final_path,
            media_type="image/jpeg",
            filename="enhanced_plate_like.jpg"
        )

    except Exception as e:
        tb = traceback.format_exc()
        return {"detail": str(e), "traceback": tb}
