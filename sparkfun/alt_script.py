#!/usr/bin/env python3
import sys
import time

try:
    import qwiic_as7265x
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sparkfun-qwiic-as7265x"])
    import qwiic_as7265x

print("Initializing AS7265x sensor...")
sensor = qwiic_as7265x.QwiicAS7265x()

if sensor.begin() != 0:
    print("ERROR: Sensor not detected!")
    sys.exit(1)

print("Sensor initialized!\n")

# Configuration
sensor.disable_indicator()
sensor.set_gain(3)  # 64x gain
sensor.set_integration_cycles(49)
sensor.set_measurement_mode(2)  # Mode 2: CONTINUOUS (more reliable than one-shot)

time.sleep(0.5)

# Enable LEDs
sensor.enable_bulb(sensor.kLedWhite)
sensor.enable_bulb(sensor.kLedIr)
sensor.enable_bulb(sensor.kLedUv)
time.sleep(0.2)

print("Taking measurement...")
sensor.take_measurements()

# Calculate expected wait time based on integration cycles
# Formula: time = 2.8ms * (cycles + 1) * 3 sensors
wait_time = (2.8 * (49 + 1) * 3) / 1000
print(f"Waiting {wait_time:.2f}s for measurement to complete...")
time.sleep(wait_time + 0.5)  # Add 0.5s buffer

print("\n" + "="*60)
print("SPECTRAL READING")
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

for wavelength, value in channels:
    print(f"{wavelength:15s}: {value:8.2f}")

print("="*60)
print(f"Temperature: {sensor.get_temperature_average():.1f}°C\n")

# Cleanup
sensor.disable_bulb(sensor.kLedWhite)
sensor.disable_bulb(sensor.kLedIr)
sensor.disable_bulb(sensor.kLedUv)

print("Done!")
