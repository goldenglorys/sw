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
sensor = qwiic_as7265x.QwiicAS7265x()

# Initialize
if sensor.begin() != 0:
    print("\nERROR: Sensor not detected!")
    print("Check your wiring and run: sudo i2cdetect -y -r 1")
    sys.exit(1)

print("Sensor initialized successfully!\n")

# Configure sensor - KEY CHANGES HERE
sensor.disable_indicator()  # Turn off LED indicator to reduce power draw
sensor.disable_bulb()  # Start with bulb off
sensor.set_gain(3)  # 64x gain (0=1x, 1=3.7x, 2=16x, 3=64x)
sensor.set_integration_cycles(49)  # Integration time
sensor.set_measurement_mode(3)  # All channels, one-shot

# CRITICAL: Add delay after configuration
time.sleep(0.5)

print("Taking measurement...")

# Enable bulb for measurement (if measuring reflectance)
sensor.enable_bulb()
time.sleep(0.1)  # Let bulb stabilize

# Take measurements
sensor.take_measurements_with_bulb()  # Use this instead of take_measurements()

# Wait for data with better timeout handling
print("Waiting for data...")
timeout = 0
max_timeout = 200  # Increased timeout (20 seconds)

while not sensor.data_available():
    time.sleep(0.1)
    timeout += 1
    if timeout % 10 == 0:  # Print progress every second
        print(f"Waiting... {timeout/10:.0f}s")
    
    if timeout > max_timeout:
        print("\nTimeout waiting for data")
        print("Try:")
        print("1. Lower integration_cycles (try 10-20)")
        print("2. Check if sensor firmware is up to date")
        print("3. Try measurement_mode 2 instead of 3")
        sys.exit(1)

print("\n" + "="*50)
print("SPECTRAL READING")
print("="*50)

# Read all channels
channels = [
    ("410nm (A)", sensor.get_calibrated_A()),
    ("435nm (B)", sensor.get_calibrated_B()),
    ("460nm (C)", sensor.get_calibrated_C()),
    ("485nm (D)", sensor.get_calibrated_D()),
    ("510nm (E)", sensor.get_calibrated_E()),
    ("535nm (F)", sensor.get_calibrated_F()),
    ("560nm (G)", sensor.get_calibrated_G()),
    ("585nm (H)", sensor.get_calibrated_H()),
    ("610nm (R)", sensor.get_calibrated_R()),
    ("645nm (I)", sensor.get_calibrated_I()),
    ("680nm (S)", sensor.get_calibrated_S()),
    ("705nm (J)", sensor.get_calibrated_J()),
    ("730nm (T)", sensor.get_calibrated_T()),
    ("760nm (U)", sensor.get_calibrated_U()),
    ("810nm (V)", sensor.get_calibrated_V()),
    ("860nm (W)", sensor.get_calibrated_W()),
    ("900nm (K)", sensor.get_calibrated_K()),
    ("940nm (L)", sensor.get_calibrated_L()),
]

for wavelength, value in channels:
    print(f"{wavelength}: {value:.2f}")

print("="*50)
print(f"\nTemperature: {sensor.get_temperature():.1f}°C")
print("\nSUCCESS! Sensor is working!\n")

# Turn off bulb
sensor.disable_bulb()
