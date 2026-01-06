#!/usr/bin/env python3
"""
Collect NIR spectral data for ML model training
"""
import sys
import csv
import time
from datetime import datetime
from nir_sensor import NIRSensor


def collect_dataset():
    """Interactive data collection for building training dataset"""
    
    print("=" * 60)
    print("NIR SPECTRAL DATA COLLECTOR")
    print("=" * 60)
    
    # Initialize sensor
    sensor = NIRSensor()
    
    # Setup CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"nir_dataset_{timestamp}.csv"
    
    # CSV header: wavelengths + temp + metadata
    header = [f"{wl}nm" for wl in sensor.WAVELENGTHS] + [
        'temperature', 'material_type', 'condition', 'notes', 'sample_id'
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        sample_count = 0
        
        print(f"\nLogging to: {csv_file}")
        print("\nMaterial types: PET, HDPE, PP, LDPE, PVC, PS, Aluminum, Glass, Other")
        print("Condition: clean, dirty, degraded, mixed")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                # Get metadata
                material = input("Material type: ").strip().upper()
                if not material:
                    continue
                
                condition = input("Condition (clean/dirty/degraded): ").strip().lower()
                notes = input("Notes (optional): ").strip()
                
                # Take multiple readings
                n_readings = int(input("Number of readings (default 5): ") or "5")
                
                print(f"\nTaking {n_readings} readings...")
                for i in range(n_readings):
                    print(f"  Reading {i+1}/{n_readings}...", end=" ")
                    
                    data = sensor.take_measurement()
                    sample_count += 1
                    
                    # Write to CSV
                    row = data['values'] + [
                        data['temperature'],
                        material,
                        condition,
                        notes,
                        f"{material}_{sample_count:04d}"
                    ]
                    writer.writerow(row)
                    f.flush()  # Save immediately
                    
                    print("✓")
                    time.sleep(0.5)
                
                print(f"\nTotal samples collected: {sample_count}")
                print("-" * 60)
        
        except KeyboardInterrupt:
            print(f"\n\nCollection stopped. Total samples: {sample_count}")
            print(f"Data saved to: {csv_file}")
        
        finally:
            sensor.close()


if __name__ == "__main__":
    collect_dataset()
