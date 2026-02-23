import cv2
import os
import csv
import sys
import time
import requests
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image
from gradio_client import Client, handle_file
from pyzbar.pyzbar import decode


USE_CLOUD_INFERENCE = True  # Set to False to use local TinyYolo
CLOUD_SPACE_ID = "fotonlink/barcode-detector"  # HF Space ID
MOTION_SENSITIVITY = 25  # Lower = More sensitive (0-255)
MOTION_AREA_THRESHOLD = 500  # How many pixels must change to trigger
UPLOAD_COOLDOWN = 2.0  # Seconds to wait between cloud uploads


class CameraDetector:
    def __init__(self, camera_source=0):
        """Initialize detector with Cloud, Motion, and CSV capabilities"""
        self.platform = self._detect_platform()
        print(f"Platform: {self.platform}")

        # --- 1. SETUP INFERENCE ENGINE ---
        if USE_CLOUD_INFERENCE:
            print("Mode: CLOUD SENTINEL")
            print(f"Target: {CLOUD_SPACE_ID}")
            try:
                self.client = Client(CLOUD_SPACE_ID)
                self.model = None
                print("✓ Connected to Hugging Face Cloud")
            except Exception as e:
                print(f"FATAL: Could not connect to Cloud: {e}")
                sys.exit(1)
        else:
            print("Mode: LOCAL INFERENCE")
            try:
                from model.tiny_yolo import TinyYolo

                self.model = TinyYolo(classes=2)
                print("Local TinyYolo Model loaded")
            except Exception as e:
                print(f"FATAL: Could not load local model: {e}")
                sys.exit(1)

        # --- 2. SETUP CAMERA ---
        if camera_source == 0:
            camera_source = self._get_camera_source()

        print(f"Opening camera: {camera_source}")
        self.camera = self._init_camera(camera_source)

        if not self.camera.isOpened():
            print("FATAL: Camera failed to open.")
            sys.exit(1)

        # Verify read
        ret, _ = self.camera.read()
        if not ret:
            print("FATAL: Camera opened but returned empty frame.")
            sys.exit(1)

        # --- 3. STATE VARIABLES ---
        self.temp_dir = tempfile.mkdtemp()
        self.seen_codes = set()
        self.frame_count = 0
        self.prev_gray = None  # For motion detection
        self.last_upload_time = 0

    def _detect_platform(self):
        """Auto-detect platform (Jetson, Pi, Desktop)"""
        if os.path.exists("/etc/nv_tegra_release"):
            return "jetson"
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    if "raspberry pi" in f.read().lower():
                        return "rpi"
            except:
                pass
        return "desktop"

    def _get_camera_source(self):
        """Get GStreamer pipeline for Jetson or default index"""
        if self.platform == "jetson":
            return (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
                "framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
            )
        return 0

    def _init_camera(self, source):
        if self.platform == "jetson" and isinstance(source, str):
            return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(source)

    def _detect_motion(self, frame):
        """
        Returns True if significant motion is detected compared to previous frame.
        Used to trigger cloud inference.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        # Compute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)[
            1
        ]

        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Update reference frame slightly (rolling background update)
        # We blend the new frame into the old one so slow changes don't trigger
        cv2.accumulateWeighted(gray, self.prev_gray.astype("float"), 0.5)
        self.prev_gray = cv2.convertScaleAbs(self.prev_gray.astype("float"))

        # Check if enough pixels changed
        cnts, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for c in cnts:
            if cv2.contourArea(c) > MOTION_AREA_THRESHOLD:
                motion_detected = True
                break

        return motion_detected

    def _cloud_inference(self, frame):
        """Send frame to Hugging Face via Gradio Client"""
        # Save to temp file
        temp_path = os.path.join(self.temp_dir, "query.jpg")
        cv2.imwrite(temp_path, frame)

        try:
            # Call /run_inference endpoint
            result = self.client.predict(
                img=handle_file(temp_path), api_name="/run_inference"
            )
            return result
        except Exception as e:
            print(f"Cloud Error: {e}")
            return None

    def _local_inference(self, frame):
        """
        Local YOLO + PyZbar detection (your existing code)

        Returns same format as cloud
        """

        # YOLO detection
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yolo_result = self.model.predict(pil_image)

        # PyZbar decoding
        decoded_list = decode(pil_image)

        code_data = None
        code_type = None

        if decoded_list:
            obj = decoded_list[0]
            code_type = obj.type

            try:
                code_data = obj.data.decode('utf-8', errors='ignore')
            except:
                code_data = str(obj.data)

            if code_data:
                code_data = code_data.strip()

        # Product lookup
        product_name = "N/A"
        if code_data and code_data.isdigit():
            product_name = self._get_product_info(code_data)

        return {
            "status": "success" if code_data else "no_detection",
            "code_data": code_data,
            "code_type": code_type,
            "product_name": product_name,
            "confidence": 1.0 if code_data else 0.0
        }

    def _get_product_info(self, barcode):
        """Query OpenFoodFacts"""
        try:
            url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
            r = requests.get(url, timeout=2)
            data = r.json()
            if data.get("status") == 1:
                return data["product"].get("product_name", "Unknown Product")
        except:
            pass
        return "Product not found"

    def run(self, display=True, save_detections=True, log_csv=True):
        """Main Loop: Motion Trigger -> Cloud -> CSV/Image Log"""

        # --- SETUP LOGGING ---
        csv_file = None
        csv_writer = None
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        images_dir = log_dir / "images"
        images_dir.mkdir(exist_ok=True)

        if log_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / "csvs"
            csv_path.mkdir(exist_ok=True)
            csv_filename = csv_path / f"log_{timestamp}.csv"

            try:
                csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(["Timestamp", "Data", "Type", "Product", "Method"])
                print(f"Logging to: {csv_filename}")
            except Exception as e:
                print(f"Error opening CSV: {e}")

        print("\n=== SYSTEM ARMED ===")
        print("Waiting for motion to trigger cloud scan...")
        print("Press 'q' to quit.\n")

        try:
            while True:
                self.frame_count += 1
                ret, frame = self.camera.read()
                if not ret:
                    break

                # --- 1. MOTION DETECTION TRIGGER ---
                should_scan = False

                # Check for motion
                motion = self._detect_motion(frame)

                # Logic: If motion detected AND cooldown passed -> Scan
                current_time = time.time()
                if motion and (current_time - self.last_upload_time > UPLOAD_COOLDOWN):
                    should_scan = True
                    self.last_upload_time = current_time
                    # Visual Indicator: Blue border = Uploading
                    if display:
                        cv2.rectangle(
                            frame,
                            (0, 0),
                            (frame.shape[1], frame.shape[0]),
                            (255, 0, 0),
                            10,
                        )

                # --- 2. INFERENCE (Cloud or Local) ---
                result = None
                if should_scan:
                    if USE_CLOUD_INFERENCE:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Motion! Sending to Cloud..."
                        )
                        result = self._cloud_inference(frame)
                    else:
                        print(
                            f"💻 [{datetime.now().strftime('%H:%M:%S')}] Motion! Local Scan..."
                        )
                        result = self._local_inference(frame)

                # --- 3. PROCESS RESULTS ---
                if result and result.get("status") == "success":
                    code_data = result.get("code_data")
                    code_type = result.get("code_type")
                    product = result.get("product_name", "N/A")

                    # Check uniqueness
                    if code_data and code_data not in self.seen_codes:
                        self.seen_codes.add(code_data)
                        print(f"NEW CODE: {code_data} ({product})")

                        # A. Log to CSV
                        if csv_writer:
                            csv_writer.writerow(
                                [
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    code_data,
                                    code_type,
                                    product,
                                    "cloud" if USE_CLOUD_INFERENCE else "local",
                                ]
                            )
                            csv_file.flush()

                        # B. Save Image
                        if save_detections:
                            safe_name = "".join(c for c in code_data if c.isalnum())[
                                :20
                            ]
                            fname = (
                                images_dir
                                / f"{safe_name}_{datetime.now().strftime('%H%M%S')}.jpg"
                            )
                            cv2.imwrite(str(fname), frame)
                            print(f"   Saved image: {fname.name}")

                # --- 4. DISPLAY ---
                if display:
                    # Overlay stats
                    status_text = f"Codes: {len(self.seen_codes)}"
                    cv2.putText(
                        frame,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    if motion:  # Show red dot if motion is currently happening
                        cv2.circle(
                            frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1
                        )

                    cv2.imshow("Smart Sentry", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            if csv_file:
                csv_file.close()
            self.camera.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")


if __name__ == "__main__":
    detector = CameraDetector()
    detector.run()
