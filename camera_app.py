from pathlib import Path
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
import sys
import traceback
from sparkfun.sensor_thread import NIRSensorThread
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

os.environ["OPENCV_VIDEOIO_BACKEND"] = "opencv_egl"


class CameraDetector:
    def __init__(self, camera_source=0):
        """Initialize camera-based barcode/QR detector"""
        self.platform = self._detect_platform()
        print(f"Platform: {self.platform}")

        print("Loading YOLO model...")
        self.model = TinyYolo(classes=2)
        print("Model loaded successfully")

        if camera_source == 0:
            camera_source = self._get_camera_source()

        print(f"Attempting to open camera with GStreamer pipeline:\n{camera_source}\n")
        self.camera = self._init_camera(camera_source)

        if not self.camera.isOpened():
            print("\n" + "#" * 50)
            print("### FATAL: camera.isOpened() returned FALSE. ###")
            print("### Review the GStreamer logs above for the root cause. ###")
            print("#" * 50 + "\n")
            sys.exit(1)

        print("SUCCESS: camera.isOpened() returned TRUE.")

        ret, test_frame = self.camera.read()
        if not ret or test_frame is None:
            print("ERROR: Camera opened but cannot read frames!")
            sys.exit(1)

        print(f"Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")

        self.seen_codes = set()
        self.frame_count = 0
        self.nir_thread = NIRSensorThread(interval_seconds=2.0)
        self._nir_available = self.nir_thread.start()
        if self._nir_available:
            print("NIR sensor running in background")
        else:
            print("NIR sensor not available — camera-only mode")

    def _detect_platform(self):
        """Auto-detect platform, with Docker override"""
        platform_override = os.environ.get("PLATFORM_OVERRIDE")
        if platform_override:
            print(f"Platform override detected: {platform_override}")
            return platform_override.lower()
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

    # def _get_camera_source(self):
    #     """Get appropriate camera source"""
    #     if self.platform == "jetson":
    #         return (
    #             "nvarguscamerasrc ! "
    #             "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
    #             "nvvidconv ! "
    #             "video/x-raw, format=BGRx ! "
    #             "videoconvert ! "
    #             "video/x-raw, format=BGR ! "
    #             "appsink drop=true max-buffers=1"
    #         )
    #     return 0

    def _get_camera_source(self):
        """Get appropriate camera source via platform_config (respects FORCE_USB)"""
        from platform_config import get_camera_source

        return get_camera_source(self.platform)

    # def _init_camera(self, source):
    #     """Initialize camera"""
    #     try:
    #         if self.platform == "jetson" and isinstance(source, str):
    #             return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    #         return cv2.VideoCapture(source)
    #     except Exception as e:
    #         print(f"ERROR initializing camera: {e}")
    #         sys.exit(1)

    def _init_camera(self, source):
        """Initialize camera"""
        try:
            from platform_config import get_opencv_backend

            if isinstance(source, str):
                return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
            backend = get_opencv_backend(self.platform)
            return (
                cv2.VideoCapture(source, backend)
                if backend
                else cv2.VideoCapture(source)
            )
        except Exception as e:
            print(f"ERROR initializing camera: {e}")
            sys.exit(1)

    def _get_product_info(self, barcode):
        """Query OpenFoodFacts API"""
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
        """Main detection loop"""

        # Initialize CSV
        csv_file = None
        csv_writer = None
        if log_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            csvs_dir = log_dir / "csvs"
            csvs_dir.mkdir(exist_ok=True)
            csv_filename = csvs_dir / f"barcode_log_{timestamp}.csv"
            try:
                csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                try:
                    nir_headers = self.nir_thread.get_csv_headers()
                except Exception as e:
                    print(f"WARNING: Could not get NIR headers: {e}")
                    nir_headers = [
                        f"NIR_{wl}nm"
                        for wl in [
                            410,
                            435,
                            460,
                            485,
                            510,
                            535,
                            560,
                            585,
                            610,
                            645,
                            680,
                            705,
                            730,
                            760,
                            810,
                            860,
                            900,
                            940,
                        ]
                    ] + ["NIR_Temperature", "NIR_Timestamp", "NIR_ReadingIndex"]
                csv_writer.writerow(
                    [
                        "Timestamp",
                        "Code Data",
                        "Code Type",
                        "Product Name",
                        "Platform",
                        "Detected Class",
                        "Confidence",
                    ]
                    + nir_headers
                )
                csv_file.flush()
                print(f"Logging to: {csv_filename}\n")
            except Exception as e:
                print(f"WARNING: Could not open CSV: {e}")
                log_csv = False

        detection_count = 0
        platform_name = platform_module.system()
        consecutive_failures = 0
        max_failures = 10

        print("Starting barcode/QR detection. Press 'q' to quit.\n")

        try:
            while True:
                self.frame_count += 1

                # Read frame
                try:
                    ret, frame = self.camera.read()
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("Too many failures, exiting")
                        break
                    continue

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("Too many failures, exiting")
                        break
                    continue

                consecutive_failures = 0

                # Convert to RGB
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except:
                    continue

                # YOLO prediction (for visualization only)
                try:
                    annotated = self.model.predict_array(frame_rgb)
                except Exception as e:
                    annotated = frame_rgb

                # Convert to BGR
                try:
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                except:
                    annotated_bgr = frame

                # PyZbar decoding (actual detection)
                barcode_data = None
                code_type = "UNKNOWN"
                detected_class = "N/A"
                confidence = "N/A"

                try:
                    pil_image = Image.fromarray(frame_rgb)
                    decoded_list = decode(pil_image)

                    if decoded_list:
                        decoded_obj = decoded_list[0]
                        code_type = decoded_obj.type

                        try:
                            barcode_data = decoded_obj.data.decode(
                                "utf-8", errors="ignore"
                            )
                        except:
                            barcode_data = str(decoded_obj.data)

                        # Clean data
                        if barcode_data:
                            barcode_data = barcode_data.strip()
                            barcode_data = "".join(
                                char
                                for char in barcode_data
                                if ord(char) >= 32 or char == "\n"
                            )

                        # Use PyZbar type as detected class (more accurate than YOLO's 2 classes)
                        detected_class = code_type
                        confidence = "1.00"  # PyZbar either decodes it or doesn't

                except:
                    pass

                # Get product info
                product_name = "N/A"
                if barcode_data and barcode_data.isdigit():
                    product_name = self._get_product_info(barcode_data)

                # Check if new
                is_new_code = False
                if barcode_data and barcode_data not in self.seen_codes:
                    self.seen_codes.add(barcode_data)
                    detection_count += 1
                    is_new_code = True

                # Display
                if display:
                    try:
                        nir_data = self.nir_thread.get_latest()
                        nir_status = (
                            f"NIR #{nir_data['reading_index']}"
                            if nir_data
                            else "NIR: waiting..."
                        )
                        cv2.putText(
                            annotated_bgr,
                            f"Codes: {detection_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            annotated_bgr,
                            nir_status,
                            (10, 130),  # below the barcode text lines
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 200, 0),  # yellow-ish colour
                            2,
                        )

                        if barcode_data:
                            color = (0, 255, 0) if is_new_code else (0, 165, 255)
                            display_data = (
                                barcode_data[:50] + "..."
                                if len(barcode_data) > 50
                                else barcode_data
                            )

                            cv2.putText(
                                annotated_bgr,
                                f"{code_type}: {display_data}",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )
                            if not is_new_code:
                                cv2.putText(
                                    annotated_bgr,
                                    "(Already Scanned)",
                                    (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 165, 255),
                                    2,
                                )

                        cv2.imshow("Barcode/QR Detection", annotated_bgr)
                    except:
                        pass

                # Log to CSV
                if log_csv and barcode_data and is_new_code and csv_writer:
                    try:
                        nir_reading = self.nir_thread.get_latest()
                        try:
                            nir_row = self.nir_thread.get_csv_row(nir_reading)
                        except Exception as e:
                            print(f"WARNING: NIR row error: {e}")
                            nir_row = [""] * 21
                        csv_writer.writerow(
                            [
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                barcode_data,
                                code_type,
                                product_name,
                                platform_name,
                                detected_class,
                                confidence,
                            ]
                            + nir_row
                        )
                        csv_file.flush()
                    except Exception as e:
                        print(f"CSV write error: {e}")
                    else:
                        print(f"  -> CSV row written")

                # Console log
                if barcode_data and is_new_code:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] {code_type}: {barcode_data}"
                    )

                # Save image
                if save_detections and barcode_data and is_new_code:
                    try:
                        safe_data = "".join(
                            c for c in barcode_data if c.isalnum() or c in "-_."
                        )[:30]
                        # filename = f'detection_{safe_data}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                        # cv2.imwrite(filename, annotated_bgr)
                        images_dir = log_dir / "images"
                        images_dir.mkdir(exist_ok=True)
                        filename = (
                            images_dir
                            / f'detection_{safe_data}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                        )
                        cv2.imwrite(
                            str(filename), annotated_bgr
                        )  # ✅ Saves to logs/images/
                        print(f"  → Saved: {filename}")
                    except:
                        pass

                # Quit
                if display:
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            print("\nQuitting...")
                            break
                    except:
                        pass

        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("\nCleaning up...")
            if csv_file:
                try:
                    csv_file.close()
                    print("CSV file closed")
                except:
                    pass

            try:
                self.nir_thread.stop()
            except:
                pass

            try:
                self.camera.release()
                print("Camera released")
            except:
                pass

            if display:
                try:
                    cv2.destroyAllWindows()
                    print("Windows closed")
                except:
                    pass

            print(f"\nSession complete!")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Unique codes: {detection_count}")


if __name__ == "__main__":
    if os.path.exists("/opt/homebrew/lib"):
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

    try:
        detector = CameraDetector()
        detector.run()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
