#!/usr/bin/env python3
"""
NIR Sensor Background Thread
Runs SparkFun AS7265x continuously in a separate thread so it doesn't
block the camera detection loop. The latest reading is always available.

Place this file in: sparkfun/sensor_thread.py
"""

import threading
import time
import sys
from datetime import datetime

try:
    from sparkfun.nir_sensor import NIRSensor
except ImportError:
    try:
        from nir_sensor import NIRSensor
    except ImportError:
        NIRSensor = None


class NIRSensorThread:
    """
    Runs NIR sensor in a background thread.
    Camera loop grabs latest reading via .get_latest() without blocking.

    Usage:
        nir_thread = NIRSensorThread()
        nir_thread.start()

        # In your camera loop:
        reading = nir_thread.get_latest()
        if reading:
            print(reading['values'])  # 18 spectral values

        nir_thread.stop()
    """

    def __init__(self, interval_seconds=2.0, gain=3, integration_cycles=49):
        """
        Args:
            interval_seconds: How often to take a new reading (default: 2s)
                              Set higher if you want less frequent captures.
                              Note: each measurement itself takes ~0.5-1s.
            gain: Sensor gain (0=1x, 1=3.7x, 2=16x, 3=64x)
            integration_cycles: Sensor integration (0-255, default 49)
        """
        self.interval = interval_seconds
        self.gain = gain
        self.integration_cycles = integration_cycles

        self._sensor = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_reading = None  # Most recent successful reading
        self._reading_count = 0
        self._error_count = 0
        self.available = False  # True once sensor is confirmed working

    def start(self):
        """Initialize sensor and start background thread."""
        if NIRSensor is None:
            print("WARNING: NIR sensor library not available. Sensor thread disabled.")
            return False

        try:
            print("Initializing NIR sensor...")
            self._sensor = NIRSensor(
                gain=self.gain, integration_cycles=self.integration_cycles
            )
            self.available = True
            print("NIR sensor ready. Starting background thread...")

            self._thread = threading.Thread(
                target=self._run_loop,
                name="NIRSensorThread",
                daemon=True,  # Dies automatically when main program exits
            )
            self._thread.start()
            return True

        except Exception as e:
            print(f"WARNING: Could not initialize NIR sensor: {e}")
            print("         Running without NIR data. Check I2C wiring.")
            self.available = False
            return False

    def _run_loop(self):
        """Background loop - takes readings continuously."""
        while not self._stop_event.is_set():
            try:
                data = self._sensor.take_measurement(with_leds=True)

                # Add extra metadata
                data["iso_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ]
                data["reading_index"] = self._reading_count

                with self._lock:
                    self._latest_reading = data
                    self._reading_count += 1

                # Wait before next reading (minus the time already spent measuring)
                self._stop_event.wait(timeout=max(0.1, self.interval - 1.0))

            except Exception as e:
                self._error_count += 1
                print(f"NIR sensor error #{self._error_count}: {e}")
                if self._error_count > 10:
                    print("Too many NIR errors. Stopping sensor thread.")
                    break
                self._stop_event.wait(timeout=2.0)

    def get_latest(self):
        """
        Get the most recent NIR reading (non-blocking).

        Returns:
            dict or None: {
                'wavelengths': [410, ..., 940],
                'values': [float x 18],
                'temperature': float,
                'timestamp': float,
                'iso_timestamp': str,
                'reading_index': int
            }
            Returns None if no reading available yet.
        """
        with self._lock:
            return self._latest_reading.copy() if self._latest_reading else None

    def get_csv_headers(self):
        """Returns list of column names for CSV logging."""
        from sparkfun.nir_sensor import NIRSensor as NS

        wavelengths = NS.WAVELENGTHS
        return [f"NIR_{wl}nm" for wl in wavelengths] + [
            "NIR_Temperature",
            "NIR_Timestamp",
            "NIR_ReadingIndex",
        ]

    def get_csv_row(self, reading=None):
        """
        Returns list of values matching get_csv_headers().
        Pass a specific reading, or leave None to use latest.
        """
        if reading is None:
            reading = self.get_latest()

        if reading is None:
            # No data yet - fill with empty strings
            n_cols = 18 + 3  # 18 wavelengths + temp + timestamp + index
            return [""] * n_cols

        return reading["values"] + [
            f"{reading['temperature']:.1f}",
            reading["iso_timestamp"],
            reading["reading_index"],
        ]

    def stop(self):
        """Stop the background thread and clean up."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._sensor:
            try:
                self._sensor.close()
            except:
                pass
        print(f"NIR sensor stopped. Total readings: {self._reading_count}")

    @property
    def reading_count(self):
        return self._reading_count


if __name__ == "__main__":
    print("Testing NIR sensor thread...")

    nir = NIRSensorThread(interval_seconds=3.0)
    if not nir.start():
        print("Could not start sensor. Exiting.")
        sys.exit(1)

    print("Sensor running. Taking 3 readings (press Ctrl+C to stop early)...\n")

    try:
        for i in range(3):
            time.sleep(3.5)
            reading = nir.get_latest()
            if reading:
                print(
                    f"Reading #{reading['reading_index']} @ {reading['iso_timestamp']}"
                )
                for wl, val in zip(reading["wavelengths"], reading["values"]):
                    print(f"  {wl}nm: {val:.2f}")
                print(f"  Temp: {reading['temperature']:.1f}°C\n")
            else:
                print("No reading yet...\n")
    except KeyboardInterrupt:
        pass
    finally:
        nir.stop()
