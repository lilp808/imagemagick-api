from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import uuid
import os
import pytesseract
import traceback 

import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.post("/ocr_image")
async def ocr_image(file: UploadFile = File(...)):
    try:
        # 1️⃣ Save uploaded file
        input_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 2️⃣ Load image with OpenCV
        img = cv2.imread(input_path)
        if img is None:
            os.remove(input_path)
            return JSONResponse({"detail": "Cannot read image"}, status_code=400)

        # 3️⃣ Convert to gray and optional preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        # 4️⃣ OCR with pytesseract
        text = pytesseract.image_to_string(gray, lang='eng+tha')

        # Clean up
        os.remove(input_path)

        return JSONResponse({"text": text})

    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Error: %s", tb)
        return JSONResponse({"detail": str(e), "traceback": tb}, status_code=500)
