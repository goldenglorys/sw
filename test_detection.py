#!/usr/bin/env python3
"""
Testing script for Barcode/QR Detection System
Tests model loading, camera access, and code detection
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
import io

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages are installed"""
    print("=" * 60)
    print("TEST 1: Package Imports")
    print("=" * 60)
    
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'pyzbar': 'PyZbar',
        'numpy': 'NumPy',
        'requests': 'Requests'
    }
    
    all_passed = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_platform_detection():
    """Test platform detection"""
    print("\n" + "=" * 60)
    print("TEST 2: Platform Detection")
    print("=" * 60)
    
    try:
        from platform_config import detect_platform, get_camera_source
        
        platform = detect_platform()
        camera_source = get_camera_source()
        
        print(f"  ✓ Platform detected: {platform}")
        print(f"  ✓ Camera source: {camera_source}")
        return True
    except Exception as e:
        print(f"  ✗ Platform detection failed: {e}")
        print("  Note: platform_config.py may not exist yet")
        return False


def test_model_loading():
    """Test YOLO model loading"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Loading")
    print("=" * 60)
    
    try:
        from model.tiny_yolo import TinyYolo
        
        print("  Loading model...")
        model = TinyYolo(classes=2)
        print("  ✓ Model loaded successfully")
        print(f"  ✓ Model input size: {model.model_size}x{model.model_size}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def test_camera_access():
    """Test camera access"""
    print("\n" + "=" * 60)
    print("TEST 4: Camera Access")
    print("=" * 60)
    
    try:
        # Try to detect platform
        try:
            from platform_config import detect_platform, get_camera_source
            platform = detect_platform()
            camera_source = get_camera_source(platform)
        except:
            platform = 'desktop'
            camera_source = 0
        
        print(f"  Platform: {platform}")
        print(f"  Attempting to open camera: {camera_source}")
        
        if platform == 'jetson':
            cap = cv2.VideoCapture(camera_source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("  ✗ Failed to open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("  ✗ Camera opened but failed to read frame")
            cap.release()
            return False
        
        print(f"  ✓ Camera working")
        print(f"  ✓ Frame size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  ✓ Frame channels: {frame.shape[2]}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"  ✗ Camera test failed: {e}")
        return False


def test_barcode_decoding():
    """Test barcode/QR decoding with synthetic image"""
    print("\n" + "=" * 60)
    print("TEST 5: Barcode/QR Decoding")
    print("=" * 60)
    
    try:
        from barcode import get_barcode
        import qrcode
        
        # Generate a test QR code
        print("  Generating test QR code...")
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data("https://github.com/test")
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to BytesIO
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        # Test decoding
        print("  Decoding test QR code...")
        code_data, code_type, product_info = get_barcode(buf)
        
        if code_data:
            print(f"  ✓ QR code decoded successfully")
            print(f"  ✓ Type: {code_type}")
            print(f"  ✓ Data: {code_data}")
            return True
        else:
            print("  ✗ Failed to decode QR code")
            print("  Note: Make sure pyzbar is installed correctly")
            return False
            
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Install with: pip install qrcode[pil]")
        return False
    except Exception as e:
        print(f"  ✗ Decoding test failed: {e}")
        return False


def test_csv_logging():
    """Test CSV file creation and writing"""
    print("\n" + "=" * 60)
    print("TEST 6: CSV Logging")
    print("=" * 60)
    
    try:
        import csv
        from datetime import datetime
        
        test_file = "test_detections.csv"
        
        # Write test data
        with open(test_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Code Data', 'Code Type', 'Product Name', 'Platform'])
            writer.writerow([
                datetime.now().isoformat(),
                '1234567890123',
                'EAN13',
                'Test Product',
                'test'
            ])
        
        # Read back
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) == 2:
            print(f"  ✓ CSV file created and written")
            print(f"  ✓ Test file: {test_file}")
            os.remove(test_file)
            print(f"  ✓ Test file cleaned up")
            return True
        else:
            print(f"  ✗ CSV file has unexpected content")
            return False
            
    except Exception as e:
        print(f"  ✗ CSV logging test failed: {e}")
        return False


def test_model_inference():
    """Test model inference with random image"""
    print("\n" + "=" * 60)
    print("TEST 7: Model Inference")
    print("=" * 60)
    
    try:
        from model.tiny_yolo import TinyYolo
        
        print("  Loading model...")
        model = TinyYolo(classes=2)
        
        # Create random test image
        print("  Creating test image...")
        test_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Test prediction
        print("  Running inference...")
        result = model.predict_array(test_image)
        
        if result is not None and result.shape == test_image.shape:
            print(f"  ✓ Inference successful")
            print(f"  ✓ Output shape: {result.shape}")
            return True
        else:
            print(f"  ✗ Inference returned unexpected result")
            return False
            
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        return False


def test_api_connection():
    """Test API connection"""
    print("\n" + "=" * 60)
    print("TEST 8: API Connection")
    print("=" * 60)
    
    try:
        import requests
        
        # Test with a known barcode (Nutella)
        test_barcode = "3017620422003"
        url = f"https://world.openfoodfacts.org/api/v0/product/{test_barcode}.json"
        
        print(f"  Testing API with barcode: {test_barcode}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 1:
                product_name = data.get('product', {}).get('product_name', 'N/A')
                print(f"  ✓ API connection successful")
                print(f"  ✓ Product found: {product_name}")
                return True
            else:
                print(f"  ✓ API reachable but product not found")
                return True
        else:
            print(f"  ✗ API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ API connection failed: {e}")
        print("  Note: Internet connection may be required")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Barcode/QR Detection System - Testing Suite             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Platform Detection", test_platform_detection),
        ("Model Loading", test_model_loading),
        ("Camera Access", test_camera_access),
        ("Barcode/QR Decoding", test_barcode_decoding),
        ("CSV Logging", test_csv_logging),
        ("Model Inference", test_model_inference),
        ("API Connection", test_api_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())