from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import uuid
import os
from pathlib import Path
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="License Plate Detection API", version="2.0")

# Initialize YOLO model (loaded once at startup)
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
model = None

@app.on_event("startup")
async def load_model():
    """Load YOLO model on startup"""
    global model
    try:
        logger.info(f"Loading YOLO model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        # Warm up the model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "License Plate Detection API", "version": "2.0", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "model_loaded": model is not None}


def enhance_plate_image(img_array: np.ndarray, min_width: int = 400) -> np.ndarray:
    """
    Enhanced image processing pipeline for license plate clarity
    
    Args:
        img_array: Input image as numpy array (BGR)
        min_width: Minimum width to upscale to
        
    Returns:
        Enhanced image as numpy array (BGR)
    """
    # Convert to PIL for better processing
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Upscale if too small
    width, height = pil_img.size
    if width < min_width:
        scale_factor = min_width / width
        new_size = (int(width * scale_factor), int(height * scale_factor))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert back to OpenCV for CLAHE
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise while preserving edges (bilateral filter)
    img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # Convert back to PIL for sharpening
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Unsharp mask for better text clarity
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)
    
    # Convert back to BGR for OpenCV
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return final_img


def detect_and_crop_plates(image_path: str, conf_threshold: float = 0.25) -> List[np.ndarray]:
    """
    Detect license plates using YOLOv8 and crop them
    
    Args:
        image_path: Path to input image
        conf_threshold: Confidence threshold for detection
        
    Returns:
        List of cropped and enhanced plate images
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run YOLO detection
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
        classes=[0, 1, 2, 3, 5, 7]  # Common vehicle classes that might have plates
    )
    
    cropped_plates = []
    
    # Alternative: Use license plate specific detection
    # For better results, detect all objects and filter by aspect ratio
    result = results[0]
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # If no detections, try to detect rectangular regions
    if len(result.boxes) == 0:
        logger.warning("No objects detected by YOLO, trying edge detection fallback")
        # Fallback to edge-based detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = w_c / h_c if h_c > 0 else 0
            area = w_c * h_c
            
            # License plates typically have aspect ratio 2-5 and reasonable size
            if 1.5 <= aspect_ratio <= 6 and 1000 <= area <= (h * w * 0.3):
                # Add padding
                padding = 0.15
                x1 = max(0, int(x - w_c * padding))
                y1 = max(0, int(y - h_c * padding))
                x2 = min(w, int(x + w_c * (1 + padding)))
                y2 = min(h, int(y + h_c * (1 + padding)))
                
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    enhanced = enhance_plate_image(crop)
                    cropped_plates.append(enhanced)
    else:
        # Process YOLO detections
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate dimensions
            box_w = x2 - x1
            box_h = y2 - y1
            aspect_ratio = box_w / box_h if box_h > 0 else 0
            
            # Filter by aspect ratio (license plates are wider than tall)
            if aspect_ratio < 1.2:
                continue
            
            # Add padding (15%)
            padding = 0.15
            x1_padded = max(0, int(x1 - box_w * padding))
            y1_padded = max(0, int(y1 - box_h * padding))
            x2_padded = min(w, int(x2 + box_w * padding))
            y2_padded = min(h, int(y2 + box_h * padding))
            
            # Crop the plate region
            crop = img[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if crop.size > 0:
                # Enhance the cropped plate
                enhanced = enhance_plate_image(crop)
                cropped_plates.append(enhanced)
    
    # Sort by confidence (if available) or by y-coordinate
    # For now, we'll just return as-is
    
    return cropped_plates


def merge_plates_vertically(plates: List[np.ndarray], spacing: int = 20, border: int = 10) -> np.ndarray:
    """
    Merge multiple plate images into a single vertical stack
    
    Args:
        plates: List of plate images (numpy arrays)
        spacing: Vertical spacing between plates
        border: Border around the final image
        
    Returns:
        Merged image as numpy array
    """
    if not plates:
        raise ValueError("No plates to merge")
    
    # Find maximum width
    max_width = max(plate.shape[1] for plate in plates)
    
    # Calculate total height
    total_height = sum(plate.shape[0] for plate in plates) + spacing * (len(plates) - 1) + border * 2
    
    # Create white background
    merged = np.ones((total_height, max_width + border * 2, 3), dtype=np.uint8) * 255
    
    # Place each plate
    current_y = border
    for plate in plates:
        h, w = plate.shape[:2]
        
        # Center horizontally
        x_offset = (max_width - w) // 2 + border
        
        # Place plate
        merged[current_y:current_y + h, x_offset:x_offset + w] = plate
        
        # Add border around each plate
        cv2.rectangle(merged, 
                     (x_offset - 2, current_y - 2), 
                     (x_offset + w + 2, current_y + h + 2), 
                     (200, 200, 200), 2)
        
        current_y += h + spacing
    
    return merged


@app.post("/detect-plates")
async def detect_plates(file: UploadFile = File(...)):
    """
    Detect license plates from a single image
    
    Args:
        file: Input image file
        
    Returns:
        Combined image with all detected plates
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Detect and crop plates
        plates = detect_and_crop_plates(input_path, conf_threshold=0.25)
        
        if not plates:
            raise HTTPException(status_code=404, detail="No license plates detected in the image")
        
        logger.info(f"Detected {len(plates)} license plate(s)")
        
        # Merge plates
        final_image = merge_plates_vertically(plates, spacing=20, border=10)
        
        # Save result
        output_path = f"/tmp/result_{uuid.uuid4()}.jpg"
        cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Cleanup input
        os.remove(input_path)
        
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="detected_plates.jpg",
            background=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect-plates-multi")
async def detect_plates_multi(files: List[UploadFile] = File(...)):
    """
    Detect license plates from multiple images
    
    Args:
        files: List of input image files
        
    Returns:
        Combined image with all detected plates from all images
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    all_plates = []
    temp_files = []
    
    try:
        # Process each image
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
            
            # Save uploaded file
            input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
            temp_files.append(input_path)
            
            with open(input_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"Processing image: {file.filename}")
            
            # Detect and crop plates
            plates = detect_and_crop_plates(input_path, conf_threshold=0.25)
            all_plates.extend(plates)
        
        if not all_plates:
            raise HTTPException(status_code=404, detail="No license plates detected in any image")
        
        logger.info(f"Detected {len(all_plates)} license plate(s) from {len(files)} image(s)")
        
        # Merge all plates
        final_image = merge_plates_vertically(all_plates, spacing=20, border=10)
        
        # Save result
        output_path = f"/tmp/result_{uuid.uuid4()}.jpg"
        cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="detected_plates_multi.jpg",
            background=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
