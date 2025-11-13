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

print("Initializing SparkFun AS7265x sensor...")

# Create sensor object
sensor = qwiic_as7265x.QwiicAs7265x()

# Initialize
if sensor.begin() != 0:
    print("\nERROR: Sensor not detected!")
    print("Check your wiring and run: sudo i2cdetect -y -r 1")
    sys.exit(1)

print("Sensor initialized successfully!\n")

# Configure sensor
sensor.set_gain(3)  # 64x gain
sensor.set_integration_cycles(49)
sensor.set_measurement_mode(3)  # All channels

print("Taking measurement...")
sensor.take_measurements()

# Wait for data
timeout = 0
while not sensor.data_available():
    time.sleep(0.1)
    timeout += 1
    if timeout > 100:
        print("Timeout waiting for data")
        sys.exit(1)

print("\n" + "="*50)
print("SPECTRAL READING")
print("="*50)

# Read all channels
channels = [
    ("410nm", sensor.get_calibrated_A()),
    ("435nm", sensor.get_calibrated_B()),
    ("460nm", sensor.get_calibrated_C()),
    ("485nm", sensor.get_calibrated_D()),
    ("510nm", sensor.get_calibrated_E()),
    ("535nm", sensor.get_calibrated_F()),
    ("560nm", sensor.get_calibrated_G()),
    ("585nm", sensor.get_calibrated_H()),
    ("610nm", sensor.get_calibrated_R()),
    ("645nm", sensor.get_calibrated_I()),
    ("680nm", sensor.get_calibrated_S()),
    ("705nm", sensor.get_calibrated_J()),
    ("730nm", sensor.get_calibrated_T()),
    ("760nm", sensor.get_calibrated_U()),
    ("810nm", sensor.get_calibrated_V()),
    ("860nm", sensor.get_calibrated_W()),
    ("900nm", sensor.get_calibrated_K()),
    ("940nm", sensor.get_calibrated_L()),
]

for wavelength, value in channels:
    print(f"{wavelength}: {value:.2f}")

print("="*50)
print("\nSUCCESS! Sensor is working!\n")
