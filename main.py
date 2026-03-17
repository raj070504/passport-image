import cv2
import numpy as np
import mediapipe as mp
import os
from io import BytesIO
from PIL import Image
import pillow_heif
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- App Initialization ---
# Register HEIF opener for Apple formats
pillow_heif.register_heif_opener()

app = FastAPI(title="Passport Photo Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (Optional for local testing) ---
class ImageRequest(BaseModel):
    filepath: str

# --- Helper Logic ---
def load_image_from_path(filepath: str) -> np.ndarray:
    """Reads an image directly from the local file system path."""
    if not os.path.exists(filepath):
        raise ValueError(f"File not found at {filepath}")

    file_extension = filepath.split('.')[-1].lower()
    
    if file_extension in ['heic', 'heif']:
        pil_image = Image.open(filepath)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image_matrix = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image_matrix is None:
            raise ValueError("Could not decode image. Check file format or corruption.")
        return image_matrix


# --- Core Logic ---
def remove_background(image_matrix: np.ndarray) -> np.ndarray:
    """Isolates the subject and replaces the background with pure white."""
    mp_selfie = mp.solutions.selfie_segmentation
    
    with mp_selfie.SelfieSegmentation(model_selection=0) as selfie_seg:
        h, w, _ = image_matrix.shape
        image_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)

        # Mask Generation
        seg_results = selfie_seg.process(image_rgb)
        mask = seg_results.segmentation_mask

        # Erode and Blur for clean edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Alpha Blending
        mask_3d = np.stack((mask,) * 3, axis=-1).astype(np.float32)
        image_float = image_matrix.astype(np.float32)
        white_bg_float = np.ones((h, w, 3), dtype=np.float32) * 255.0

        no_bg_float = (image_float * mask_3d) + (white_bg_float * (1.0 - mask_3d))
        return no_bg_float.astype(np.uint8)


def crop_to_passport(image_matrix: np.ndarray, final_w: int = 413, final_h: int = 531) -> np.ndarray:
    """Detects a face, centers it, and crops/pads to dynamically requested dimensions."""
    mp_face = mp.solutions.face_detection
    
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:
        h, w, _ = image_matrix.shape
        image_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        
        face_results = face_det.process(image_rgb)

        if not face_results.detections:
            raise ValueError("No face detected in the image.")

        detection = face_results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        fx, fy = int(bboxC.xmin * w), int(bboxC.ymin * h)
        fw, fh = int(bboxC.width * w), int(bboxC.height * h)

        # Calculate dynamic aspect ratio from requested width/height
        aspect_ratio = final_w / final_h

        # Smart Padding & Centering based on the new aspect ratio
        target_h = int(fh * 2.2) 
        target_w = int(target_h * aspect_ratio) 

        cx, cy = fx + (fw // 2), fy + (fh // 2)

        y1, x1 = cy - (target_h // 2), cx - (target_w // 2)
        y2, x2 = y1 + target_h, x1 + target_w

        passport_canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        img_y1, img_y2 = max(0, y1), min(h, y2)
        img_x1, img_x2 = max(0, x1), min(w, x2)

        can_y1, can_x1 = max(0, -y1), max(0, -x1)
        can_y2, can_x2 = can_y1 + (img_y2 - img_y1), can_x1 + (img_x2 - img_x1)

        passport_canvas[can_y1:can_y2, can_x1:can_x2] = image_matrix[img_y1:img_y2, img_x1:img_x2]

        # Final Resize using user-defined dimensions
        final_image = cv2.resize(passport_canvas, (final_w, final_h), interpolation=cv2.INTER_AREA)

        return final_image


# --- API Endpoints ---
@app.get("/")
def health_check():
    """Returns a welcome message to confirm the API is live."""
    return {"message": "Welcome! The Passport Photo API is live and ready. 3.0"}

async def process_upload(file: UploadFile, processing_func, **kwargs):
    """Helper function to read upload, convert to OpenCV, process, and stream back."""
    contents = await file.read()
    
    # Use PIL to safely open and convert image formats (handles HEIC via pillow_heif)
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    image_matrix = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Pass any extra arguments (like width/height) to the pipeline function
    final_matrix = processing_func(image_matrix, **kwargs)

    success, encoded_image = cv2.imencode(".jpg", final_matrix)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")

    return StreamingResponse(
        BytesIO(encoded_image.tobytes()),
        media_type="image/jpeg"
    )

@app.post("/remove-bg")
async def api_remove_bg(file: UploadFile = File(...)):
    """Accepts an uploaded image file, removes background, and streams back."""
    try:
        return await process_upload(file, remove_background)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crop")
async def api_crop(
    file: UploadFile = File(...),
    width: int = Query(413, description="Target width in pixels (Default is Indian Passport)"),
    height: int = Query(531, description="Target height in pixels (Default is Indian Passport)")
):
    """Accepts an uploaded pre-processed image, detects face, crops to requested size, and streams back."""
    try:
        return await process_upload(file, crop_to_passport, final_w=width, final_h=height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def api_generate_full(
    file: UploadFile = File(...),
    width: int = Query(413, description="Target width in pixels (Default is Indian Passport)"),
    height: int = Query(531, description="Target height in pixels (Default is Indian Passport)")
):
    """Runs the full pipeline: background removal followed by face cropping."""
    def full_pipeline(image_matrix, final_w, final_h):
        no_bg = remove_background(image_matrix)
        return crop_to_passport(no_bg, final_w, final_h)
        
    try:
        return await process_upload(file, full_pipeline, final_w=width, final_h=height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))