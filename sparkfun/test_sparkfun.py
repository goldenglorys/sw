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

# Configure sensor
sensor.disable_indicator()  # Turn off status LED
sensor.set_gain(3)  # 64x gain (0=1x, 1=3.7x, 2=16x, 3=64x)
sensor.set_integration_cycles(49)  # Integration time
sensor.set_measurement_mode(3)  # Mode 3: All channels, one-shot

# Add delay after configuration
time.sleep(0.5)

print("Taking measurement...")

# Enable all three bulbs (White, IR, UV)
# According to docs: kLedWhite=0x00, kLedIr=0x01, kLedUv=0x02
sensor.enable_bulb(sensor.kLedWhite)  # Enable white LED
sensor.enable_bulb(sensor.kLedIr)     # Enable IR LED
sensor.enable_bulb(sensor.kLedUv)     # Enable UV LED
time.sleep(0.2)  # Let bulbs stabilize

# Take measurements - use the simpler method
sensor.take_measurements()

# Wait for data with better feedback
print("Waiting for data...")
timeout = 0
max_timeout = 200  # 20 seconds

while not sensor.data_available():
    time.sleep(0.1)
    timeout += 1
    if timeout % 10 == 0:  # Print progress every second
        print(f"  Still waiting... {timeout/10:.0f}s")
    
    if timeout > max_timeout:
        print("\nTimeout waiting for data!")
        print("\nTroubleshooting tips:")
        print("1. Try lower integration_cycles: sensor.set_integration_cycles(10)")
        print("2. Try measurement_mode 2 (continuous): sensor.set_measurement_mode(2)")
        print("3. Check power supply - sensor needs stable 3.3V")
        
        # Turn off bulbs before exiting
        sensor.disable_bulb(sensor.kLedWhite)
        sensor.disable_bulb(sensor.kLedIr)
        sensor.disable_bulb(sensor.kLedUv)
        sys.exit(1)

print("✓ Data ready!\n")

print("="*60)
print("SPECTRAL READING FROM AS7265x TRIAD")
print("="*60)

# Read all 18 channels
channels = [
    ("410nm (A - UV)", sensor.get_calibrated_a()),
    ("435nm (B - UV)", sensor.get_calibrated_b()),
    ("460nm (C - UV)", sensor.get_calibrated_c()),
    ("485nm (D - UV)", sensor.get_calibrated_d()),
    ("510nm (E - UV)", sensor.get_calibrated_e()),
    ("535nm (F - UV)", sensor.get_calibrated_f()),
    ("560nm (G - Vis)", sensor.get_calibrated_g()),
    ("585nm (H - Vis)", sensor.get_calibrated_h()),
    ("610nm (R - Vis)", sensor.get_calibrated_r()),
    ("645nm (I - Vis)", sensor.get_calibrated_i()),
    ("680nm (S - Vis)", sensor.get_calibrated_s()),
    ("705nm (J - Vis)", sensor.get_calibrated_j()),
    ("730nm (T - NIR)", sensor.get_calibrated_t()),
    ("760nm (U - NIR)", sensor.get_calibrated_u()),
    ("810nm (V - NIR)", sensor.get_calibrated_v()),
    ("860nm (W - NIR)", sensor.get_calibrated_w()),
    ("900nm (K - NIR)", sensor.get_calibrated_k()),
    ("940nm (L - NIR)", sensor.get_calibrated_l()),
]

for wavelength, value in channels:
    print(f"{wavelength:20s}: {value:8.2f}")

print("="*60)
print(f"\nAverage Temperature: {sensor.get_temperature_average():.1f}°C")
print("\nSUCCESS! Sensor is working properly!\n")

# Turn off all bulbs
sensor.disable_bulb(sensor.kLedWhite)
sensor.disable_bulb(sensor.kLedIr)
sensor.disable_bulb(sensor.kLedUv)
