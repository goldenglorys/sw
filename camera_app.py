import cv2
from model.tiny_yolo import TinyYolo
from PIL import Image
import os
import csv
from datetime import datetime
import platform as platform_module
from pyzbar.pyzbar import decode
import requests
import json
from platform_config import detect_platform, get_camera_source, get_opencv_backend


class CameraDetector:
    def __init__(self, camera_source=None, platform=None):
        """
        Initialize camera-based barcode/QR detector

        Args:
            camera_source: Camera index or CSI camera string (None for auto-detect)
            platform: 'desktop', 'jetson', or 'rpi' (None for auto-detect)
        """
        # Auto-detect platform if not specified
        self.platform = platform or detect_platform()
        print(f"Platform: {self.platform}")
        
        # Load YOLO model with 2 classes
        self.model = TinyYolo(classes=2)
        
        # Auto-select camera source if not provided
        if camera_source is None:
            camera_source = get_camera_source(self.platform)
        
        # Initialize camera
        backend = get_opencv_backend(self.platform)
        if backend:
            self.camera = cv2.VideoCapture(camera_source, backend)
        else:
            self.camera = cv2.VideoCapture(camera_source)
        
        self.seen_codes = set()

    def _get_product_info(self, barcode):
        """Query OpenFoodFacts API for product information"""
        try:
            address = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
            r = requests.get(address, timeout=5)
            data = json.loads(r.text)
            if data.get("status") == 1 and data.get("product"):
                return data["product"].get("product_name", "Unknown Product")
            return "Product not found"
        except:
            return "API Error"

    def run(self, display=True, save_detections=False, log_csv=True):
        """
        Main detection loop with automatic scanning

        Args:
            display: Show live video feed with detections
            save_detections: Save detected barcode images to disk
            log_csv: Log detections to CSV file
        """
        # Initialize CSV logging
        csv_file = None
        csv_writer = None
        if log_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"barcode_log_{timestamp}.csv"
            csv_file = open(csv_filename, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Timestamp",
                "Code Data",
                "Code Type",
                "Product Name",
                "Platform",
                "YOLO Class",
                "Confidence"
            ])
            print(f"Logging to: {csv_filename}")

        detection_count = 0
        platform_name = platform_module.system()

        print("Starting automatic barcode/QR detection. Press 'q' to quit.\n")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Convert BGR to RGB for YOLO model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get YOLO predictions with metadata
                annotated, boxes, scores, classes, nums = self.model.predict_array(
                    frame_rgb, return_metadata=True
                )

                # Convert annotated image back to BGR for OpenCV display
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                # Direct pyzbar decoding on original frame
                pil_image = Image.fromarray(frame_rgb)
                decoded_list = decode(pil_image)

                barcode_data = None
                code_type = "UNKNOWN"
                yolo_class = "N/A"
                confidence = "N/A"

                # Extract YOLO metadata if detections exist
                if nums[0] > 0:
                    confidence = f"{float(scores[0][0]):.2f}"
                    class_idx = int(classes[0][0])
                    from settings import Settings
                    class_names = Settings.class_names
                    yolo_class = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

                # Process pyzbar detections
                if decoded_list:
                    decoded_obj = decoded_list[0]
                    code_type = decoded_obj.type
                    
                    # Decode barcode data
                    try:
                        barcode_data = decoded_obj.data.decode('utf-8', errors='ignore')
                    except:
                        barcode_data = str(decoded_obj.data)

                # Get product info (only for numeric barcodes)
                product_name = "N/A"
                if barcode_data and barcode_data.isdigit():
                    product_name = self._get_product_info(barcode_data)

                # Check if this is a new code
                is_new_code = False
                if barcode_data and barcode_data not in self.seen_codes:
                    self.seen_codes.add(barcode_data)
                    detection_count += 1
                    is_new_code = True

                # Add visual indicators to display
                if display:
                    # Live counter
                    cv2.putText(
                        annotated_bgr,
                        f"Codes: {detection_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    # Show code info if detected
                    if barcode_data:
                        color = (0, 255, 0) if is_new_code else (0, 165, 255)
                        info_text = f"{code_type}: {barcode_data}"
                        cv2.putText(
                            annotated_bgr,
                            info_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                        if not is_new_code:
                            cv2.putText(
                                annotated_bgr,
                                "(Already Scanned)",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),
                                2
                            )

                    cv2.imshow("Barcode/QR Detection", annotated_bgr)

                # Log to CSV (only new codes)
                if log_csv and barcode_data and is_new_code:
                    current_time = datetime.now()
                    csv_writer.writerow([
                        current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        barcode_data,
                        code_type,
                        product_name,
                        platform_name,
                        yolo_class,
                        confidence
                    ])
                    csv_file.flush()

                # Console logging
                if barcode_data and is_new_code:
                    current_time = datetime.now()
                    print(
                        f"[DETECTED] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"Type: {code_type} | Data: {barcode_data}"
                    )

                # Save detection image if requested
                if save_detections and barcode_data and is_new_code:
                    current_time = datetime.now()
                    filename = f'detection_{barcode_data}_{current_time.strftime("%Y%m%d_%H%M%S")}.jpg'
                    cv2.imwrite(filename, annotated_bgr)
                    print(f"  → Saved: {filename}")

                # Press 'q' to quit
                if display:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            if log_csv and csv_file:
                csv_file.close()
            self.camera.release()
            if display:
                cv2.destroyAllWindows()
            print(f"\nSession complete. Total unique codes: {detection_count}")


if __name__ == "__main__":
    # Set environment variable for zbar on macOS
    if os.path.exists("/opt/homebrew/lib"):
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

    # Simple usage - auto-detects everything
    detector = CameraDetector()
    detector.run()