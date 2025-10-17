# import cv2
# from model.tiny_yolo import TinyYolo
# from barcode import get_barcode
# from PIL import Image
# import io
# import os
# import csv
# from datetime import datetime
# import platform as platform_module


# class CameraDetector:
#     def __init__(self, camera_source=0, platform="desktop"):
#         """
#         Initialize camera-based barcode/QR detector

#         Args:
#             camera_source: Camera index (0 for default) or CSI camera string
#             platform: 'desktop', 'jetson', or 'rpi'
#         """
#         self.platform = platform
#         # self.model = TinyYolo()  
#         self.model = TinyYolo(classes=2)  # 2 classes: Barcode and QR
#         self.camera = self._init_camera(camera_source)
#         self.seen_codes = set()  # Track all scanned codes to prevent duplicates

#     def _init_camera(self, source):
#         """Initialize camera based on platform"""
#         if self.platform == "jetson":
#             # CSI camera on Jetson Nano
#             gst_str = (
#                 "nvarguscamerasrc ! "
#                 "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
#                 "nvvidconv ! video/x-raw, format=BGRx ! "
#                 "videoconvert ! video/x-raw, format=BGR ! appsink"
#             )
#             return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
#         elif self.platform == "rpi":
#             return cv2.VideoCapture(source)
#         else:
#             # Desktop/laptop webcam
#             return cv2.VideoCapture(source)

#     def run(self, display=True, save_detections=False, log_csv=True):
#         """
#         Main detection loop with automatic scanning

#         Args:
#             display: Show live video feed with detections
#             save_detections: Save detected barcode images to disk
#             log_csv: Log detections to CSV file
#         """

#         # Initialize CSV logging
#         csv_path = "barcode_detections.csv"
#         csv_exists = os.path.exists(csv_path)

#         if log_csv:
#             csv_file = open(csv_path, "a", newline="")
#             csv_writer = csv.writer(csv_file)

#             if not csv_exists:
#                 # Create CSV with headers
#                 csv_writer.writerow(
#                     [
#                         "Timestamp",
#                         "Code Data",
#                         "Code Type",
#                         "Product Name",
#                         "Platform",
#                         "YOLO Class",
#                         "Confidence",
#                     ]
#                 )
#                 csv_file.flush()

#         platform_name = platform_module.system()
#         detection_count = 0  # Live counter for unique detections

#         print("Starting automatic barcode/QR detection. Press 'q' to quit.")

#         try:
#             while True:
#                 ret, frame = self.camera.read()
#                 if not ret:
#                     print("Failed to grab frame")
#                     break

#                 # Perform YOLO detection on every frame
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 annotated_rgb = self.model.predict_array(frame_rgb)

#                 # Convert back to BGR for OpenCV display
#                 annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

#                 # Barcode/QR decoding using pyzbar
#                 pil_img = Image.fromarray(frame_rgb)
#                 buf = io.BytesIO()
#                 pil_img.save(buf, format="PNG")
#                 buf.seek(0)

#                 barcode_data, info = get_barcode(buf)

#                 # Check if this is a new code
#                 is_new_code = False
#                 if barcode_data and barcode_data not in self.seen_codes:
#                     self.seen_codes.add(barcode_data)
#                     detection_count += 1
#                     is_new_code = True

#                 # Display frame with annotations
#                 if display:
#                     display_frame = annotated_bgr.copy()

#                     # Draw live counter in top-left corner
#                     cv2.putText(
#                         display_frame,
#                         f"Codes Scanned: {detection_count}",
#                         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 255, 0),
#                         2,
#                     )

#                     # If barcode/QR detected, show info
#                     if barcode_data:
#                         # Determine code type from pyzbar
#                         code_type = "UNKNOWN"
#                         if info and "type" in info:
#                             code_type = info["type"]

#                         # Display code data and type
#                         text = f"{code_type}: {barcode_data}"
#                         cv2.putText(
#                             display_frame,
#                             text,
#                             (10, 70),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (
#                                 (0, 255, 0) if is_new_code else (0, 165, 255)
#                             ),  # Green for new, orange for duplicate
#                             2,
#                         )

#                         # Show duplicate status
#                         if not is_new_code:
#                             cv2.putText(
#                                 display_frame,
#                                 "(Already Scanned)",
#                                 (10, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.6,
#                                 (0, 165, 255),
#                                 2,
#                             )
#                     else:
#                         cv2.putText(
#                             display_frame,
#                             "Scanning...",
#                             (10, 70),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (0, 0, 255),
#                             2,
#                         )

#                     cv2.imshow("Barcode/QR Detection", display_frame)

#                 # Log to CSV and console (only for NEW codes)
#                 if barcode_data and is_new_code and log_csv:
#                     current_time = datetime.now()

#                     # Get product info if available
#                     product_name = "N/A"
#                     if info and info.get("status"):
#                         product_name = info.get("product", {}).get(
#                             "product_name", "N/A"
#                         )

#                     # Get code type from pyzbar
#                     code_type = info.get("type", "UNKNOWN") if info else "UNKNOWN"

#                     # Determine YOLO class (would need to modify predict_array to return this)
#                     yolo_class = "Barcode/QR"  # Placeholder
#                     confidence = "N/A"  # Placeholder

#                     # Write to CSV
#                     csv_writer.writerow(
#                         [
#                             current_time.strftime("%Y-%m-%d %H:%M:%S"),
#                             barcode_data,
#                             code_type,
#                             product_name,
#                             platform_name,
#                             yolo_class,
#                             confidence,
#                         ]
#                     )
#                     csv_file.flush()

#                     # Console output with [DETECTED] prefix
#                     print(
#                         f"[DETECTED] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
#                         f"Type: {code_type} | Data: {barcode_data} | Product: {product_name}"
#                     )

#                 # Console output for duplicates (not saved to CSV)
#                 elif barcode_data and not is_new_code:
#                     current_time = datetime.now()
#                     code_type = info.get("type", "UNKNOWN") if info else "UNKNOWN"
#                     print(
#                         f"[DUPLICATE] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
#                         f"Type: {code_type} | Data: {barcode_data}"
#                     )

#                 # Save detection image if requested (only for new codes)
#                 if save_detections and barcode_data and is_new_code:
#                     current_time = datetime.now()
#                     filename = f'detection_{barcode_data}_{current_time.strftime("%Y%m%d_%H%M%S")}.jpg'
#                     cv2.imwrite(filename, annotated_bgr)
#                     print(f"  → Saved image: {filename}")

#                 # Press 'q' to quit
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord("q"):
#                     break

#         finally:
#             if log_csv:
#                 csv_file.close()
#             self.camera.release()
#             cv2.destroyAllWindows()
#             print(f"\nSession complete. Total unique codes scanned: {detection_count}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Camera-based barcode/QR detection")
#     parser.add_argument(
#         "--platform",
#         choices=["desktop", "jetson", "rpi"],
#         default="desktop",
#         help="Target platform",
#     )
#     parser.add_argument(
#         "--camera", type=int, default=0, help="Camera index (0 for default)"
#     )
#     parser.add_argument(
#         "--no-display", action="store_true", help="Run headless (no GUI)"
#     )
#     parser.add_argument("--save", action="store_true", help="Save detection images")

#     args = parser.parse_args()

#     # Set environment variable for zbar on macOS/Linux
#     if os.path.exists("/opt/homebrew/lib"):
#         os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

#     detector = CameraDetector(camera_source=args.camera, platform=args.platform)
#     detector.run(display=not args.no_display, save_detections=args.save)

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
  
  
class CameraDetector:  
    def __init__(self, camera_source=0, platform="desktop"):  
        """  
        Initialize camera-based barcode/QR detector  
  
        Args:  
            camera_source: Camera index (0 for default) or CSI camera string  
            platform: 'desktop', 'jetson', or 'rpi'  
        """  
        self.platform = platform
        # self.model = TinyYolo()
        self.model = TinyYolo(classes=2)  # 2 classes: Barcode and QR  
        self.camera = self._init_camera(camera_source)  
        self.seen_codes = set()  # Track all scanned codes to prevent duplicates  
  
    def _init_camera(self, source):  
        """Initialize camera based on platform"""  
        if self.platform == "jetson":  
            # CSI camera on Jetson Nano  
            gst_str = (  
                "nvarguscamerasrc ! "  
                "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "  
                "nvvidconv ! video/x-raw, format=BGRx ! "  
                "videoconvert ! video/x-raw, format=BGR ! appsink"  
            )  
            return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)  
        elif self.platform == "rpi":  
            return cv2.VideoCapture(source)  
        else:  
            # Desktop/laptop webcam  
            return cv2.VideoCapture(source)  
  
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
                    # Get first detection's confidence and class  
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
                    # Live counter in top-left  
                    cv2.putText(  
                        annotated_bgr,  
                        f"Barcodes: {detection_count}",  
                        (10, 30),  
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        1,  
                        (0, 255, 0),  
                        2  
                    )  
  
                    # Show code info if detected  
                    if barcode_data:  
                        # Color: green for new, orange for duplicate  
                        color = (0, 255, 0) if is_new_code else (0, 165, 255)  
                          
                        # Display code type and data  
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
  
                        # Show duplicate indicator  
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
                if barcode_data:  
                    current_time = datetime.now()  
                    if is_new_code:  
                        print(  
                            f"[DETECTED] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "  
                            f"Type: {code_type} | Data: {barcode_data}"  
                        )  
                    else:  
                        print(  
                            f"[DUPLICATE] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "  
                            f"Type: {code_type} | Data: {barcode_data}"  
                        )  
  
                # Save detection image if requested (only for new codes)  
                if save_detections and barcode_data and is_new_code:  
                    current_time = datetime.now()  
                    filename = f'detection_{barcode_data}_{current_time.strftime("%Y%m%d_%H%M%S")}.jpg'  
                    cv2.imwrite(filename, annotated_bgr)  
                    print(f"  → Saved image: {filename}")  
  
                # Press 'q' to quit  
                if display:  
                    key = cv2.waitKey(1) & 0xFF  
                    if key == ord("q"):  
                        break  
  
        finally:  
            if log_csv and csv_file:  
                csv_file.close()  
            self.camera.release()  
            if display:  
                cv2.destroyAllWindows()  
            print(f"\nSession complete. Total unique codes scanned: {detection_count}")  
  
  
if __name__ == "__main__":  
    import argparse  
  
    parser = argparse.ArgumentParser(description="Camera-based barcode/QR detection")  
    parser.add_argument(  
        "--platform",  
        choices=["desktop", "jetson", "rpi"],  
        default="desktop",  
        help="Target platform",  
    )  
    parser.add_argument(  
        "--camera", type=int, default=0, help="Camera index (0 for default)"  
    )  
    parser.add_argument(  
        "--no-display", action="store_true", help="Run headless (no GUI)"  
    )  
    parser.add_argument("--save", action="store_true", help="Save detection images")  
  
    args = parser.parse_args()  
  
    # Set environment variable for zbar on macOS/Linux  
    if os.path.exists("/opt/homebrew/lib"):  
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"  
  
    detector = CameraDetector(camera_source=args.camera, platform=args.platform)  
    detector.run(display=not args.no_display, save_detections=args.save)