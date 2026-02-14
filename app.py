import gradio as gr
import cv2
import numpy as np
import onnxruntime as ort
from pyzbar.pyzbar import decode
from PIL import Image
import requests

# 1. Load ONNX model directly
MODEL_PATH = "yolo_barcode_detector.onnx" 
try:
    # Use CPU specifically to avoid the CUDA errors in logs
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX Model loaded successfully on CPU")
except Exception as e:
    print(f"FAILED to load ONNX: {e}")

def get_product_info(barcode):
    try:
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("status") == 1:
            return data["product"].get("product_name", "Unknown Product")
    except: pass
    return "Not found"

def run_inference(img):
    """
    Main function: Process image for YOLO, decode data with PyZbar, 
    and lookup product info.
    """
    if img is None: 
        return {"status": "error", "message": "No image provided"}
    
    # --- STEP 1: YOLO DETECTION ---
    # 1. Resize to the exact 416x416 the model expects
    input_img = cv2.resize(img, (416, 416))
    
    # 2. Normalize to [0, 1] as per local training settings
    input_img = input_img.astype(np.float32) / 255.0
    
    # 3. Add Batch Dimension (Shape: [1, 416, 416, 3])
    # Note: We do NOT transpose because the model expects Channels-Last (HWC)
    input_img = np.expand_dims(input_img, axis=0)

    try:
        # Run the ONNX session
        outputs = session.run(None, {session.get_inputs()[0].name: input_img})
        # outputs[0] usually contains the detection boxes/scores
        detections = outputs[0] 
    except Exception as e:
        print(f"Inference Error: {e}")
        detections = None

    # --- STEP 2: PYZBAR DECODING ---
    # Convert numpy array to PIL for PyZbar
    pil_img = Image.fromarray(img)
    decoded_list = decode(pil_img)
    
    # Initialize the response dictionary in the expected format
    res = {
        "status": "no_detection",
        "code_data": None,
        "code_type": None,
        "product_name": "N/A",
        "confidence": 0.0
    }

    if decoded_list:
        # Take the first decoded object
        obj = decoded_list[0]
        try:
            data = obj.data.decode('utf-8', errors='ignore').strip()
        except:
            data = str(obj.data)
            
        # Update results
        res.update({
            "status": "success",
            "code_data": data,
            "code_type": obj.type,
            "confidence": 1.0  # PyZbar is binary (works or doesn't)
        })
        
        # --- STEP 3: PRODUCT LOOKUP ---
        # Only lookup if data is numeric (standard for barcodes)
        if data.isdigit():
            res["product_name"] = get_product_info(data)
    
    return res

demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="numpy"),
    outputs=gr.JSON(),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()