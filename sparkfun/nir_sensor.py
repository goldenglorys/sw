#!/usr/bin/env python3
"""
Production-ready NIR sensor interface for SparkFun AS7265x
Based on alt_script.py (the one that actually works!)
"""
import sys
import time
import numpy as np

try:
    import qwiic_as7265x
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sparkfun-qwiic-as7265x"])
    import qwiic_as7265x


class NIRSensor:
    """Interface for SparkFun AS7265x 18-channel spectral sensor"""
    
    # Wavelength mapping (nm)
    WAVELENGTHS = [
        410, 435, 460, 485, 510, 535,  # UV range (A-F)
        560, 585, 610, 645, 680, 705,  # Visible (G-J, R, I, S)
        730, 760, 810, 860, 900, 940   # NIR (T-W, K, L)
    ]
    
    CHANNEL_NAMES = [
        'A', 'B', 'C', 'D', 'E', 'F',
        'G', 'H', 'R', 'I', 'S', 'J',
        'T', 'U', 'V', 'W', 'K', 'L'
    ]
    
    def __init__(self, gain=3, integration_cycles=49):
        """
        Initialize sensor
        
        Args:
            gain: 0=1x, 1=3.7x, 2=16x, 3=64x (default: 3)
            integration_cycles: 0-255 (default: 49, ~420ms total)
        """
        self.sensor = qwiic_as7265x.QwiicAS7265x()
        self.gain = gain
        self.integration_cycles = integration_cycles
        
        if self.sensor.begin() != 0:
            raise ConnectionError("Sensor not detected! Check I2C wiring.")
        
        self._configure()
    
    def _configure(self):
        """Configure sensor with working settings"""
        self.sensor.disable_indicator()
        self.sensor.set_gain(self.gain)
        self.sensor.set_integration_cycles(self.integration_cycles)
        self.sensor.set_measurement_mode(2)  # Mode 2: CONTINUOUS (reliable)
        time.sleep(0.5)
    
    def enable_leds(self):
        """Turn on all three LED illumination sources"""
        self.sensor.enable_bulb(self.sensor.kLedWhite)
        self.sensor.enable_bulb(self.sensor.kLedIr)
        self.sensor.enable_bulb(self.sensor.kLedUv)
        time.sleep(0.2)  # Let LEDs stabilize
    
    def disable_leds(self):
        """Turn off all LEDs"""
        self.sensor.disable_bulb(self.sensor.kLedWhite)
        self.sensor.disable_bulb(self.sensor.kLedIr)
        self.sensor.disable_bulb(self.sensor.kLedUv)
    
    def take_measurement(self, with_leds=True):
        """
        Capture 18-channel spectral reading
        
        Args:
            with_leds: Enable LED illumination (default: True)
        
        Returns:
            dict: {
                'wavelengths': [410, 435, ..., 940],
                'values': [float, ...],  # 18 calibrated readings
                'temperature': float,
                'timestamp': float
            }
        """
        if with_leds:
            self.enable_leds()
        
        # Trigger measurement
        self.sensor.take_measurements()
        
        # Calculate wait time (formula from datasheet)
        wait_time = (2.8 * (self.integration_cycles + 1) * 3) / 1000
        time.sleep(wait_time + 0.5)  # Add buffer
        
        # Read all 18 channels
        readings = [
            self.sensor.get_calibrated_a(),  # 410nm
            self.sensor.get_calibrated_b(),  # 435nm
            self.sensor.get_calibrated_c(),  # 460nm
            self.sensor.get_calibrated_d(),  # 485nm
            self.sensor.get_calibrated_e(),  # 510nm
            self.sensor.get_calibrated_f(),  # 535nm
            self.sensor.get_calibrated_g(),  # 560nm
            self.sensor.get_calibrated_h(),  # 585nm
            self.sensor.get_calibrated_r(),  # 610nm
            self.sensor.get_calibrated_i(),  # 645nm
            self.sensor.get_calibrated_s(),  # 680nm
            self.sensor.get_calibrated_j(),  # 705nm
            self.sensor.get_calibrated_t(),  # 730nm
            self.sensor.get_calibrated_u(),  # 760nm
            self.sensor.get_calibrated_v(),  # 810nm
            self.sensor.get_calibrated_w(),  # 860nm
            self.sensor.get_calibrated_k(),  # 900nm
            self.sensor.get_calibrated_l(),  # 940nm
        ]
        
        temperature = self.sensor.get_temperature_average()
        
        if with_leds:
            self.disable_leds()
        
        return {
            'wavelengths': self.WAVELENGTHS,
            'values': readings,
            'temperature': temperature,
            'timestamp': time.time()
        }
    
    def take_multiple_measurements(self, n=5, delay=0.5):
        """
        Take multiple measurements and return statistics
        
        Args:
            n: Number of measurements (default: 5)
            delay: Delay between measurements in seconds (default: 0.5)
        
        Returns:
            dict: {
                'mean': array of mean values,
                'std': array of standard deviations,
                'cv': array of coefficients of variation (std/mean),
                'temperature': float
            }
        """
        measurements = []
        
        for i in range(n):
            data = self.take_measurement()
            measurements.append(data['values'])
            if i < n - 1:
                time.sleep(delay)
        
        measurements = np.array(measurements)
        mean = np.mean(measurements, axis=0)
        std = np.std(measurements, axis=0)
        cv = std / (mean + 1e-10)  # Avoid division by zero
        
        return {
            'wavelengths': self.WAVELENGTHS,
            'mean': mean.tolist(),
            'std': std.tolist(),
            'cv': cv.tolist(),
            'temperature': data['temperature']
        }
    
    def print_measurement(self, data):
        """Pretty print measurement results"""
        print("=" * 60)
        print("SPECTRAL READING")
        print("=" * 60)
        
        for wavelength, channel, value in zip(
            data['wavelengths'], 
            self.CHANNEL_NAMES, 
            data['values']
        ):
            print(f"{wavelength}nm ({channel:1s}): {value:8.2f}")
        
        print("=" * 60)
        print(f"Temperature: {data['temperature']:.1f}°C")
        print("=" * 60)
    
    def get_spectrum_vector(self, with_leds=True):
        """
        Get spectrum as numpy array (for ML model input)
        
        Returns:
            np.array: Shape (19,) = [18 spectral values + temperature]
        """
        data = self.take_measurement(with_leds=with_leds)
        return np.array(data['values'] + [data['temperature']])
    
    def close(self):
        """Cleanup - turn off LEDs"""
        self.disable_leds()


# Example usage
if __name__ == "__main__":
    print("Initializing NIR sensor...")
    
    try:
        sensor = NIRSensor()
        print("Sensor initialized successfully!\n")
        
        # Single measurement
        print("Taking measurement...")
        data = sensor.take_measurement()
        sensor.print_measurement(data)
        
        # Multiple measurements for stability check
        print("\nTaking 5 measurements to check stability...")
        stats = sensor.take_multiple_measurements(n=5)
        
        print("\nMeasurement Stability:")
        print(f"{'Wavelength':<12} {'Mean':<10} {'Std Dev':<10} {'CV%':<10}")
        print("-" * 45)
        for wl, mean, std, cv in zip(
            stats['wavelengths'], 
            stats['mean'], 
            stats['std'], 
            stats['cv']
        ):
            print(f"{wl}nm{' ':<7} {mean:<10.2f} {std:<10.3f} {cv*100:<10.2f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sensor.close()
        print("\nDone!")
