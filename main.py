import cv2
import numpy as np
import mediapipe as mp
import os
from io import BytesIO
from PIL import Image
import pillow_heif
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Register HEIF opener for Apple formats
pillow_heif.register_heif_opener()

# --- Pydantic Models ---
class ImageRequest(BaseModel):
    filepath: str

# --- App Initialization ---
app = FastAPI(title="Passport Photo Generator API")

# --- Core Logic ---
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

def generate_passport_photo(image_matrix):
    mp_selfie = mp.solutions.selfie_segmentation
    mp_face = mp.solutions.face_detection

    with mp_selfie.SelfieSegmentation(model_selection=0) as selfie_seg, \
         mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:

        h, w, _ = image_matrix.shape
        image_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)

        # --- STEP 1: Mask Generation & Refinement ---
        seg_results = selfie_seg.process(image_rgb)
        mask = seg_results.segmentation_mask  # Float mask 0.0 to 1.0

        # A. ERODE: Shrink the mask to eliminate the background "halo"
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)

        # B. BLUR: Soften the edges so they don't look artificial
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # C. ALPHA BLENDING: Smoothly merge subject with white background
        mask_3d = np.stack((mask,) * 3, axis=-1).astype(np.float32)
        image_float = image_matrix.astype(np.float32)
        white_bg_float = np.ones((h, w, 3), dtype=np.float32) * 255.0

        no_bg_float = (image_float * mask_3d) + (white_bg_float * (1.0 - mask_3d))
        no_bg_image = no_bg_float.astype(np.uint8)

        # --- STEP 2: Face Detection ---
        no_bg_rgb = cv2.cvtColor(no_bg_image, cv2.COLOR_BGR2RGB)
        face_results = face_det.process(no_bg_rgb)

        if not face_results.detections:
            raise ValueError("No face detected in the image.")

        detection = face_results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        fx, fy = int(bboxC.xmin * w), int(bboxC.ymin * h)
        fw, fh = int(bboxC.width * w), int(bboxC.height * h)

        # --- STEP 3: Smart Padding & Centering ---
        target_h = int(fh * 2.2) 
        target_w = int(target_h * (35 / 45)) 

        cx, cy = fx + (fw // 2), fy + (fh // 2)

        y1, x1 = cy - (target_h // 2), cx - (target_w // 2)
        y2, x2 = y1 + target_h, x1 + target_w

        passport_canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        img_y1, img_y2 = max(0, y1), min(h, y2)
        img_x1, img_x2 = max(0, x1), min(w, x2)

        can_y1, can_x1 = max(0, -y1), max(0, -x1)
        can_y2, can_x2 = can_y1 + (img_y2 - img_y1), can_x1 + (img_x2 - img_x1)

        passport_canvas[can_y1:can_y2, can_x1:can_x2] = no_bg_image[img_y1:img_y2, img_x1:img_x2]

        # --- STEP 4: Final Resize ---
        final_image = cv2.resize(passport_canvas, (413, 531), interpolation=cv2.INTER_AREA)

        return final_image

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Returns a welcome message to confirm the API is live."""
    return {"message": "Welcome! The Passport Photo API is live and ready."}

@app.post("/generate")
def process_image(request: ImageRequest):
    """
    Accepts a local file path, processes the image to create a passport photo, 
    and streams the resulting image back to the client.
    """
    try:
        # 1. Load the image from the provided local path
        image_matrix = load_image_from_path(request.filepath)
        
        # 2. Run the halo-killer/centering pipeline
        final_matrix = generate_passport_photo(image_matrix)
        
        # 3. Encode the OpenCV matrix to JPEG format in memory
        success, encoded_image = cv2.imencode(".jpg", final_matrix)
        if not success:
            raise ValueError("Failed to encode the processed image.")
            
        # 4. Stream the image back as a response
        io_buf = BytesIO(encoded_image.tobytes())
        return StreamingResponse(io_buf, media_type="image/jpeg")
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
