import cv2  
from model.tiny_yolo import TinyYolo  
from barcode import get_barcode  
from PIL import Image  
import io  
import os
import csv
from datetime import datetime
import platform as platform_module  
  
class CameraDetector:  
    def __init__(self, camera_source=0, platform='desktop'):  
        """  
        Initialize camera-based barcode detector  
          
        Args:  
            camera_source: Camera index (0 for default) or CSI camera string  
            platform: 'desktop', 'jetson', or 'rpi'  
        """  
        self.platform = platform  
        self.model = TinyYolo()  
        self.camera = self._init_camera(camera_source)  
          
    def _init_camera(self, source):  
        """Initialize camera based on platform"""  
        if self.platform == 'jetson':  
            # CSI camera on Jetson Nano  
            gst_str = (  
                'nvarguscamerasrc ! '  
                'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! '  
                'nvvidconv ! video/x-raw, format=BGRx ! '  
                'videoconvert ! video/x-raw, format=BGR ! appsink'  
            )  
            return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)  
        elif self.platform == 'rpi':  
            # Raspberry Pi camera (requires picamera2 for better performance)  
            # For now, use USB camera fallback  
            return cv2.VideoCapture(source)  
        else:  
            # Desktop/laptop webcam  
            return cv2.VideoCapture(source)  
      
    def run(self, display=True, save_detections=False, log_csv=True):
        """
        Main detection loop with automatic scanning
        
        Args:
            display: Show live video feed with detections
            save_detections: Save detected barcode images to disk
            log_csv: Log detections to CSV file
        """
        
        # Initialize CSV logging
        csv_path = 'barcode_detections.csv'
        csv_exists = os.path.exists(csv_path)
        
        if log_csv:
            if not csv_exists:
                # Create CSV with headers
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'barcode', 'product_name', 'platform', 'confidence'])
            
            csv_file = open(csv_path, 'a', newline='')
            csv_writer = csv.writer(csv_file)
        
        print("Starting automatic barcode detection. Press 'q' to quit.")
        platform_name = platform_module.system()  # 'Darwin', 'Linux', 'Windows'
        
        # Track last detected barcode to avoid duplicate logs
        last_barcode = None
        last_detection_time = 0
        detection_cooldown = 2.0  # seconds between logging same barcode
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Perform YOLO detection on every frame
                annotated = self.model.predict_array(frame)
                
                # Barcode decoding
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                buf.seek(0)
                
                barcode, info = get_barcode(buf)
                
                # Display annotated frame with bounding boxes
                if display:
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    
                    # Add detection status text
                    if barcode:
                        cv2.putText(display_frame, f"Barcode: {barcode}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Scanning...", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
                    
                    cv2.imshow('Barcode Detection', display_frame)
                
                # Log detection to CSV
                current_time = datetime.now()
                if barcode and log_csv:
                    # Avoid duplicate logs within cooldown period
                    if barcode != last_barcode or (current_time.timestamp() - last_detection_time) > detection_cooldown:
                        product_name = "N/A"
                        if info and info.get("status"):
                            product_name = info.get('product', {}).get('product_name', 'N/A')
                        
                        # Get confidence from YOLO detection (would need to modify predict_array to return this)
                        confidence = "N/A"  # Placeholder - see modification below
                        
                        csv_writer.writerow([
                            current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            barcode,
                            product_name,
                            platform_name,
                            confidence
                        ])
                        csv_file.flush()  # Ensure immediate write
                        
                        print(f"✓ Logged: {barcode} - {product_name}")
                        last_barcode = barcode
                        last_detection_time = current_time.timestamp()
                
                if save_detections and barcode:
                    cv2.imwrite(f'detection_{barcode}_{current_time.strftime("%Y%m%d_%H%M%S")}.jpg', 
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                
                # Press 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            if log_csv:
                csv_file.close()
            self.camera.release()
            cv2.destroyAllWindows()
  
if __name__ == '__main__':  
    import argparse  
      
    parser = argparse.ArgumentParser(description='Camera-based barcode detection')  
    parser.add_argument('--platform', choices=['desktop', 'jetson', 'rpi'],   
                       default='desktop', help='Target platform')  
    parser.add_argument('--camera', type=int, default=0,   
                       help='Camera index (0 for default)')  
    parser.add_argument('--no-display', action='store_true',   
                       help='Run headless (no GUI)')  
    parser.add_argument('--save', action='store_true',   
                       help='Save detection images')  
      
    args = parser.parse_args()  
      
    # Set environment variable for zbar on macOS/Linux  
    if os.path.exists('/opt/homebrew/lib'):  
        os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'  
      
    detector = CameraDetector(camera_source=args.camera, platform=args.platform)  
    detector.run(display=not args.no_display, save_detections=args.save)