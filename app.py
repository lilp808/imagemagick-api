from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import subprocess
import uuid
import os

app = FastAPI()

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    input_path = f"/tmp/{uuid.uuid4()}.jpg"
    output_path = f"/tmp/out_{uuid.uuid4()}.jpg"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # ImageMagick preset สำหรับ OCR
    cmd = [
        "convert", input_path,
        "-colorspace", "Gray",
        "-contrast-stretch", "0",
        "-sharpen", "0x1",
        output_path
    ]

    subprocess.run(cmd, check=True)

    return FileResponse(
        output_path,
        media_type="image/jpeg",
        filename="enhanced.jpg"
    )
