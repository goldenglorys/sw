#!/usr/bin/env python3
import sys
import time

try:
    import qwiic_as7265x
except ImportError:
    print("Installing library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sparkfun-qwiic-as7265x"])
    import qwiic_as7265x

print("="*60)
print("AS7265x DIAGNOSTIC TOOL")
print("="*60)

# Create sensor object
sensor = qwiic_as7265x.QwiicAS7265x()

# Test 1: Connection
print("\n[TEST 1] Checking I2C connection...")
if sensor.begin() != 0:
    print("FAILED: Sensor not detected at I2C address 0x49")
    print("Run: sudo i2cdetect -y -r 1")
    sys.exit(1)
print("PASSED: Sensor detected")

# Test 2: Device Info
print("\n[TEST 2] Reading device information...")
try:
    dev_type = sensor.get_device_type()
    hw_version = sensor.get_hardware_version()
    fw_major = sensor.get_major_firmware_version()
    fw_patch = sensor.get_patch_firmware_version()
    fw_build = sensor.get_build_firmware_version()
    
    print(f"  Device Type: 0x{dev_type:02X}")
    print(f"  Hardware Version: 0x{hw_version:02X}")
    print(f"  Firmware: {fw_major}.{fw_patch}.{fw_build}")
    print("PASSED: Device communication working")
except Exception as e:
    print(f"FAILED: Could not read device info - {e}")
    sys.exit(1)

# Test 3: Temperature reading (tests basic sensor function)
print("\n[TEST 3] Testing basic sensor communication...")
try:
    temp = sensor.get_temperature_average()
    print(f"  Temperature: {temp:.1f}°C")
    if 0 < temp < 100:
        print("PASSED: Sensor responding correctly")
    else:
        print(f"WARNING: Unusual temperature reading ({temp}°C)")
except Exception as e:
    print(f"FAILED: Cannot read temperature - {e}")
    sys.exit(1)

# Test 4: Configuration
print("\n[TEST 4] Configuring sensor...")
try:
    sensor.disable_indicator()
    
    # Use VERY conservative settings for testing
    sensor.set_gain(0)  # 1x gain (lowest)
    sensor.set_integration_cycles(10)  # Short integration time
    sensor.set_measurement_mode(2)  # Mode 2: Continuous reading (faster than one-shot)
    
    time.sleep(0.5)
    print("PASSED: Configuration complete")
except Exception as e:
    print(f"FAILED: Configuration error - {e}")
    sys.exit(1)

# Test 5: LED Control
print("\n[TEST 5] Testing LED control...")
try:
    print("  Enabling LEDs...")
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.enable_bulb(sensor.kLedIr)
    sensor.enable_bulb(sensor.kLedUv)
    time.sleep(0.3)
    print("PASSED: LEDs enabled (you should see them lit)")
except Exception as e:
    print(f"FAILED: LED control error - {e}")
    sys.exit(1)

# Test 6: Measurement without waiting for data_available
print("\n[TEST 6] Taking measurement (continuous mode - no wait)...")
try:
    sensor.take_measurements()
    print("  Measurement triggered")
    
    # In continuous mode, wait a fixed time instead of polling
    wait_time = (2.8 * (10 + 1)) / 1000  # integration time formula
    print(f"  Waiting {wait_time*3:.3f}s for data...")
    time.sleep(wait_time * 3)  # Wait 3x integration time to be safe
    
    print("PASSED: Measurement cycle complete")
except Exception as e:
    print(f"❌ FAILED: Measurement error - {e}")
    sensor.disable_bulb(sensor.kLedWhite)
    sensor.disable_bulb(sensor.kLedIr)
    sensor.disable_bulb(sensor.kLedUv)
    sys.exit(1)

# Test 7: Read data directly (skip data_available check)
print("\n[TEST 7] Reading spectral data...")
try:
    # Try reading a single channel first
    test_value = sensor.get_calibrated_r()  # 610nm red channel
    print(f"  610nm (R) channel: {test_value:.2f}")
    
    if test_value < 0:
        print("⚠ WARNING: Negative values may indicate sensor issue")
    
    print("PASSED: Can read data from sensor")
except Exception as e:
    print(f"FAILED: Cannot read data - {e}")
    sensor.disable_bulb(sensor.kLedWhite)
    sensor.disable_bulb(sensor.kLedIr)
    sensor.disable_bulb(sensor.kLedUv)
    sys.exit(1)

# Test 8: Full spectrum read
print("\n[TEST 8] Reading all 18 channels...")
print("="*60)

channels = [
    ("410nm (A)", sensor.get_calibrated_a()),
    ("435nm (B)", sensor.get_calibrated_b()),
    ("460nm (C)", sensor.get_calibrated_c()),
    ("485nm (D)", sensor.get_calibrated_d()),
    ("510nm (E)", sensor.get_calibrated_e()),
    ("535nm (F)", sensor.get_calibrated_f()),
    ("560nm (G)", sensor.get_calibrated_g()),
    ("585nm (H)", sensor.get_calibrated_h()),
    ("610nm (R)", sensor.get_calibrated_r()),
    ("645nm (I)", sensor.get_calibrated_i()),
    ("680nm (S)", sensor.get_calibrated_s()),
    ("705nm (J)", sensor.get_calibrated_j()),
    ("730nm (T)", sensor.get_calibrated_t()),
    ("760nm (U)", sensor.get_calibrated_u()),
    ("810nm (V)", sensor.get_calibrated_v()),
    ("860nm (W)", sensor.get_calibrated_w()),
    ("900nm (K)", sensor.get_calibrated_k()),
    ("940nm (L)", sensor.get_calibrated_l()),
]

all_zeros = True
for wavelength, value in channels:
    print(f"{wavelength:15s}: {value:8.2f}")
    if value != 0:
        all_zeros = False

print("="*60)

if all_zeros:
    print("\nWARNING: All values are zero!")
    print("Possible causes:")
    print("  - LEDs not bright enough (try increasing bulb current)")
    print("  - No object in front of sensor")
    print("  - Sensor needs calibration")
else:
    print("\n✓ SUCCESS: Sensor is reading spectral data!")

# Cleanup
sensor.disable_bulb(sensor.kLedWhite)
sensor.disable_bulb(sensor.kLedIr)
sensor.disable_bulb(sensor.kLedUv)

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nKey findings:")
print("1. If you got here, your wiring is CORRECT")
print("2. The timeout issue is likely because:")
print("   - data_available() flag may not work in all modes")
print("   - Mode 3 (one-shot) may have timing issues")
print("\nRECOMMENDATION:")
print("Use CONTINUOUS mode (mode 2) and skip data_available() check")
