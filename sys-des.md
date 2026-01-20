# Reverse Vending Machine - System Architecture & Design

## 🎯 Project Overview

**Goal:** Build an intelligent reverse vending machine (RVM) that accurately identifies and validates recyclable materials using computer vision and NIR spectroscopy.

**Value Proposition:**
- **Dual Verification**: Visual detection (barcode/QR) + Material analysis (NIR spectroscopy)
- **Fraud Prevention**: Cross-validates claimed material against actual composition
- **Contamination Detection**: Rejects items with food residue or mixed materials
- **Flexible Acceptance**: Works with or without barcodes
- **Real-time Processing**: Jetson Nano edge computing for instant feedback

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    REVERSE VENDING MACHINE                         │
│                         (Jetson Nano)                              │
└────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
  ┌─────────▼──────────┐            ┌──────────▼─────────┐
  │   VISUAL SYSTEM    │            │    NIR SYSTEM      │
  │  ═══════════════   │            │  ═══════════════   │
  │                    │            │                    │
  │  CSI Camera        │            │  SparkFun AS7265x  │
  │  (IMX219)          │            │  Spectral Sensor   │
  │                    │            │                    │
  │  GStreamer         │            │  I2C (Qwiic)       │
  │  Pipeline          │            │  18 Channels       │
  │                    │            │  410-940nm         │
  └─────────┬──────────┘            └──────────┬─────────┘
            │                                   │
            │                                   │
  ┌─────────▼──────────┐            ┌──────────▼─────────┐
  │  DETECTION LAYER   │            │  SENSING LAYER     │
  │  ═══════════════   │            │  ═══════════════   │
  │                    │            │                    │
  │  YOLO v3 Tiny      │            │  Spectral Reading  │
  │  (Object Detection)│            │  - 18 wavelengths  │
  │                    │            │  - Temperature     │
  │  PyZbar            │            │  - LED control     │
  │  (Barcode Decode)  │            │                    │
  └─────────┬──────────┘            └──────────┬─────────┘
            │                                   │
            │                                   │
  ┌─────────▼──────────┐            ┌──────────▼─────────┐
  │ IDENTIFICATION     │            │  CLASSIFICATION    │
  │ ═══════════════    │            │  ═══════════════   │
  │                    │            │                    │
  │ - Decode barcode   │            │  ML Model          │
  │ - API lookup       │            │  (Random Forest)   │
  │ - Product info     │            │                    │
  │ - Claimed material │            │  Output:           │
  │                    │            │  - Material type   │
  │                    │            │  - Confidence      │
  └─────────┬──────────┘            └──────────┬─────────┘
            │                                   │
            └──────────────┬────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   VALIDATION ENGINE     │
              │   ═══════════════════   │
              │                         │
              │  Cross-Reference:       │
              │  ├─ Barcode material    │
              │  └─ NIR material        │
              │                         │
              │  Checks:                │
              │  ├─ Material match      │
              │  ├─ Contamination       │
              │  ├─ Recyclability       │
              │  └─ Quality grade       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   DECISION & REWARD     │
              │   ═══════════════════   │
              │                         │
              │  Accept/Reject          │
              │  Calculate reward       │
              │  Log transaction        │
              │  Update database        │
              └─────────────────────────┘
```

---

## 🔧 Hardware Components

### 1. Processing Unit
- **Jetson Nano Developer Kit**
  - 4GB RAM
  - 128-core Maxwell GPU
  - Quad-core ARM Cortex-A57
  - Purpose: Real-time ML inference, camera processing

### 2. Visual Sensing
- **CSI Camera Module** (e.g., IMX219)
  - 8MP resolution
  - 1280×720 @ 60fps capability
  - Connection: CSI camera port
  - Purpose: Barcode/QR code detection

### 3. Material Sensing
- **SparkFun AS7265x Spectral Triad Sensor**
  - 18 spectral channels (410nm - 940nm)
  - 3 integrated sensors (UV, Visible, NIR)
  - I2C interface via Qwiic connector
  - Built-in White/IR/UV LEDs
  - Purpose: Material composition analysis

### 4. Connectivity
- **Qwiic Cable** (I2C)
  - Connects AS7265x to Jetson Nano GPIO
  - Pins: SDA (GPIO 3), SCL (GPIO 5), 3.3V, GND

---

## 💻 Software Stack

```
┌──────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │   camera_app.py (Main RVM Controller)           │    │
│  │   - Orchestrates both systems                   │    │
│  │   - Manages detection loop                      │    │
│  │   - Logs transactions                           │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                       │
│                                                           │
│  ┌──────────────────────┐   ┌────────────────────────┐  │
│  │  COMPUTER VISION     │   │  NIR CLASSIFICATION    │  │
│  │  ═══════════════     │   │  ═══════════════       │  │
│  │                      │   │                        │  │
│  │  model/tiny_yolo.py  │   │  nir_sensor.py         │  │
│  │  - YOLOv3 Tiny       │   │  - Sensor interface    │  │
│  │  - 2 classes         │   │  - Spectral reading    │  │
│  │    (Barcode, QR)     │   │                        │  │
│  │                      │   │  nir_classifier.py     │  │
│  │  barcode.py          │   │  - ML model            │  │
│  │  - PyZbar decode     │   │  - Feature scaling     │  │
│  │  - API lookup        │   │  - Inference           │  │
│  └──────────────────────┘   └────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│                     PLATFORM LAYER                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  platform_config.py                              │   │
│  │  - Hardware detection (Jetson/Desktop/RPi)       │   │
│  │  - Camera configuration                          │   │
│  │  - GStreamer pipeline setup                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Docker Container (Optional)                     │   │
│  │  - L4T ML base image                             │   │
│  │  - GPU access                                    │   │
│  │  - Environment isolation                         │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌─────────┐
│  USER   │
│ inserts │
│  item   │
└────┬────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    ITEM SCANNING                         │
│                                                          │
│  ┌──────────────┐              ┌──────────────┐        │
│  │   CAMERA     │              │  NIR SENSOR  │        │
│  │              │              │              │        │
│  │  Captures    │              │  Measures    │        │
│  │  video frame │              │  spectrum    │        │
│  └──────┬───────┘              └──────┬───────┘        │
│         │                              │                │
│         ▼                              ▼                │
│  ┌──────────────┐              ┌──────────────┐        │
│  │ YOLO Detect  │              │  Read 18ch   │        │
│  │ Barcode/QR   │              │  410-940nm   │        │
│  └──────┬───────┘              └──────┬───────┘        │
│         │                              │                │
│         │ Bounding box                 │ Spectral       │
│         ▼                              │ vector         │
│  ┌──────────────┐                     │                │
│  │  PyZbar      │                     │                │
│  │  Decode      │                     │                │
│  └──────┬───────┘                     │                │
│         │                              │                │
│         │ "5901234567890"              │                │
│         ▼                              │                │
│  ┌──────────────┐                     │                │
│  │ OpenFoodFacts│                     │                │
│  │ API Lookup   │                     │                │
│  └──────┬───────┘                     │                │
│         │                              │                │
│         │ Product info                 ▼                │
│         │ "PET bottle"          ┌──────────────┐       │
│         │                       │ ML Classifier│       │
│         │                       │ (Random      │       │
│         │                       │  Forest)     │       │
│         │                       └──────┬───────┘       │
│         │                              │                │
│         │                              │ "HDPE"         │
└─────────┼──────────────────────────────┼────────────────┘
          │                              │
          ▼                              ▼
     ┌────────────────────────────────────────┐
     │       VALIDATION ENGINE                │
     │                                        │
     │  Claimed Material:  PET                │
     │  Detected Material: HDPE               │
     │                                        │
     │  ❌ MISMATCH DETECTED!                 │
     │                                        │
     │  Possible reasons:                     │
     │  - Wrong barcode label                 │
     │  - Container reused                    │
     │  - Fraud attempt                       │
     │                                        │
     │  Decision: REJECT                      │
     └────────────┬───────────────────────────┘
                  │
                  ▼
            ┌──────────┐
            │  RESULT  │
            │          │
            │ "Material│
            │ mismatch:│
            │ Please   │
            │ verify"  │
            └──────────┘
```

---

## 🎯 Use Cases

### Use Case 1: Standard Acceptance (Happy Path)
```
User inserts PET water bottle
  ↓
Camera detects barcode → "Coca Cola PET bottle"
  ↓
NIR sensor reads spectrum → "PET" (95% confidence)
  ↓
Validation: ✅ Match! Material is PET, clean, recyclable
  ↓
ACCEPT → Award 10 cents → Dispense receipt
```

### Use Case 2: Fraud Detection
```
User inserts HDPE milk jug with fake PET label
  ↓
Camera detects barcode → Claims "PET"
  ↓
NIR sensor reads spectrum → "HDPE" (92% confidence)
  ↓
Validation: ❌ Mismatch! Barcode says PET but sensor says HDPE
  ↓
REJECT → Display "Material verification failed"
```

### Use Case 3: No Barcode (Generic Container)
```
User inserts unlabeled PP container
  ↓
Camera: No barcode detected
  ↓
NIR sensor reads spectrum → "PP" (88% confidence)
  ↓
Validation: ✅ Identified as PP, clean, recyclable
  ↓
ACCEPT → Award 5 cents → Update database
```

### Use Case 4: Contamination Detection
```
User inserts PET bottle with liquid residue
  ↓
Camera detects barcode → "PET bottle"
  ↓
NIR sensor reads spectrum → Anomaly in visible range (560-705nm)
  ↓
Validation: ❌ Contamination detected
  ↓
REJECT → Display "Please empty and rinse container"
```

### Use Case 5: Non-Recyclable Material
```
User inserts PVC pipe
  ↓
Camera: No barcode
  ↓
NIR sensor reads spectrum → "PVC" (90% confidence)
  ↓
Validation: ❌ PVC not accepted (non-recyclable)
  ↓
REJECT → Display "PVC not accepted at this location"
```

---

## 📊 System Specifications

### Performance Targets
- **Throughput:** 1-2 seconds per item (end-to-end)
- **Visual Detection:** 10-30 FPS real-time
- **NIR Measurement:** ~500ms per reading
- **ML Inference:** <10ms (Random Forest on CPU)
- **Overall Accuracy:** >90% (combined system)

### Component Accuracy
- **Barcode Detection:** >95% (YOLO + PyZbar)
- **Material Classification:** 70-90% (depends on training data)
- **Cross-Validation:** >98% (dual system verification)

### Supported Materials
**Primary (with ML model):**
- PET (#1) - Bottles, containers
- HDPE (#2) - Milk jugs, detergent bottles
- PVC (#3) - Pipes (typically rejected)
- LDPE (#4) - Bags, films
- PP (#5) - Yogurt cups, caps
- PS (#6) - Foam containers

**Extended:**
- Aluminum - Cans
- Glass - Bottles
- Paper/Cardboard - Packaging

---

## 🔐 Key Features

### 1. Dual Verification
- **Visual:** What the item claims to be (barcode/label)
- **Spectral:** What the item actually is (molecular composition)
- **Result:** Cross-validation prevents fraud and errors

### 2. Intelligent Acceptance
- Works WITH barcodes (fast lookup + verification)
- Works WITHOUT barcodes (NIR classification only)
- Handles ambiguous cases (user feedback)

### 3. Quality Grading
- Detects contamination (food residue, dirt)
- Identifies mixed materials (multi-layer packaging)
- Grades recyclability (virgin vs recycled plastic)
- Differential rewards based on quality

### 4. Real-Time Processing
- Jetson Nano edge computing
- No cloud dependency
- Immediate user feedback
- Offline capability

### 5. Fraud Prevention
- Detects label swapping (HDPE with PET label)
- Identifies reused containers
- Validates actual vs claimed material
- Logs suspicious patterns

---

## 📁 Project Structure

```
reverse_vending_machine/
│
├── camera_app.py              # Main RVM controller (visual system)
├── barcode.py                 # Barcode/QR decoding + API lookup
├── platform_config.py         # Hardware detection & config
├── settings.py                # YOLO model configuration
├── docker_requirements.txt    # Dependencies
├── Dockerfile                 # Container setup
│
├── model/                     # Computer Vision
│   ├── tiny_yolo.py           # YOLOv3 Tiny implementation
│   ├── architecture.py        # Neural network layers
│   ├── dataset.py             # Data loading utilities
│   └── utils.py               # Helper functions
│
├── sparkfun/                  # NIR Spectroscopy
│   ├── nir_sensor.py          # Production sensor class
│   ├── diagnostics.py         # Hardware testing
│   └── data_collector.py      # ML dataset collection
│
├── data/                      # Training Data & Models
│   ├── model/
│   │   └── yolov3_train_class2_final.weights.h5
│   ├── classes.csv            # Class labels
│   ├── train_barcode_qr.tf_record
│   └── val_barcode_qr.tf_record
│
├── preprocessing/             # Data Preparation
│   ├── prepare_dataset.py     # Parse XML annotations
│   ├── convert_to_tfrecord.py # Create TFRecord files
│   └── extract_images.py      # Extract from TFRecord
│
└── archive/                   # Training images & annotations
```

---

## 🚀 Deployment Architecture

### Option 1: Standalone RVM (Current)
```
┌────────────────────────────────────┐
│         JETSON NANO                │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   RVM Application            │ │
│  │   (camera_app.py)            │ │
│  └──────────────────────────────┘ │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   NIR Classification Model   │ │
│  │   (nir_classifier.pkl)       │ │
│  └──────────────────────────────┘ │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   Local Database             │ │
│  │   (SQLite / CSV logs)        │ │
│  └──────────────────────────────┘ │
│                                    │
│  Hardware:                         │
│  - CSI Camera                      │
│  - AS7265x Sensor (I2C)           │
│  - Display (HDMI)                  │
│  - LED indicators                  │
│  - Storage (SD card)              │
└────────────────────────────────────┘
```

### Option 2: Networked RVM (Future)
```
┌────────────────────────────────────┐
│         JETSON NANO (Edge)         │
│                                    │
│  - Real-time processing            │
│  - Local ML inference              │
│  - User interface                  │
│  - Transaction logging             │
│                                    │
└───────────────┬────────────────────┘
                │
                │ WiFi/Ethernet
                ▼
┌────────────────────────────────────┐
│         CLOUD BACKEND              │
│                                    │
│  - Centralized database            │
│  - Analytics dashboard             │
│  - Model updates (OTA)             │
│  - Reward management               │
│  - Fraud detection patterns        │
│  - Multi-location management       │
└────────────────────────────────────┘
```

---

## 🔄 Machine Learning Pipeline

### Current: Computer Vision (YOLO)
```
Training Data:
├── ~600 images (barcodes + QR codes)
├── XML annotations (bounding boxes)
└── 2 classes: Barcode, QR

Model: YOLOv3 Tiny
├── Input: 416×416 RGB image
├── Architecture: Darknet backbone
├── Output: Bounding boxes + confidence
└── Trained weights: yolov3_train_class2_final.weights.h5

Deployment:
├── Platform: Jetson Nano
├── Framework: TensorFlow 2.x
├── Inference: ~30ms per frame
└── Post-processing: PyZbar decoding
```

### Planned: NIR Classification
```
Training Data (To be collected):
├── 500-1000 samples per material
├── 18 spectral features (410-940nm)
├── Metadata: temperature, condition, brand
└── 6-8 material classes

Model: Random Forest Classifier
├── Input: 19 features (18 channels + temp)
├── Preprocessing: StandardScaler + PCA
├── Architecture: 100-200 trees
├── Output: Material class + confidence
└── Export: nir_classifier.pkl + scaler.pkl

Deployment:
├── Platform: Jetson Nano (CPU)
├── Framework: scikit-learn
├── Inference: <10ms per sample
└── Integration: nir_sensor.py wrapper
```

---

## 🎨 User Interface Flow

```
┌─────────────────────────────────────┐
│    🔵 RVM Ready - Insert Item       │
│                                     │
│    [Camera Preview]                 │
│                                     │
│    💡 Status: Waiting               │
└─────────────────────────────────────┘
                  │
                  │ User inserts item
                  ▼
┌─────────────────────────────────────┐
│    🟡 Scanning Item...              │
│                                     │
│    [Live Detection View]            │
│    ┌─────────────────┐              │
│    │  Barcode: ✓     │              │
│    │  Material: ⏳   │              │
│    └─────────────────┘              │
└─────────────────────────────────────┘
                  │
                  │ Processing complete
                  ▼
┌─────────────────────────────────────┐
│    🟢 Item Accepted!                │
│                                     │
│    📦 PET Bottle (Clean)            │
│    🏆 Reward: $0.10                 │
│    ♻️  Total Today: 12 items        │
│                                     │
│    [Print Receipt] [Next Item]     │
└─────────────────────────────────────┘
```

---

## 🛡️ Security & Validation

### Validation Rules
```python
def validate_item(barcode_material, nir_material, nir_confidence, spectrum):
    # Rule 1: Material Match
    if barcode_material != nir_material:
        if nir_confidence > 0.85:
            return REJECT("Material mismatch - high confidence")
    
    # Rule 2: Contamination Check
    if detect_contamination(spectrum):
        return REJECT("Contamination detected")
    
    # Rule 3: Recyclability
    if nir_material not in ACCEPTED_MATERIALS:
        return REJECT("Material not accepted")
    
    # Rule 4: Quality Grade
    quality = assess_quality(spectrum)
    if quality < MINIMUM_QUALITY:
        return REJECT("Quality below threshold")
    
    return ACCEPT(material=nir_material, quality=quality)
```

### Fraud Prevention Scenarios
1. **Label Swapping:** Barcode says PET, NIR says HDPE → Reject
2. **Reused Container:** Barcode present but material degraded → Reduce reward
3. **Mixed Materials:** Spectral anomalies detected → Reject
4. **Non-Recyclable:** PVC detected → Reject with explanation

---

## 🏁 Summary

**This reverse vending machine combines:**
- 🎥 Computer vision (existing, working)
- 🌈 NIR spectroscopy (hardware ready, ML in progress)
- 🤖 Edge AI processing (Jetson Nano)
- ✅ Dual verification (fraud prevention)
- ♻️ Real-world recyclability validation

**Current Status:**
- Visual system: ✅ Operational
- NIR sensor: ✅ Hardware working
- NIR ML model: ⏳ Data collection phase
- Integration: ⏳ In progress

**Next Steps:**
1. Collect NIR training data (500-1000 samples)
2. Train Random Forest classifier
3. Integrate both systems
4. Field testing & validation
5. Deployment to production RVM

---

*This architecture enables a fraud-resistant, intelligent recycling system that goes beyond simple barcode scanning to provide true material verification.*
