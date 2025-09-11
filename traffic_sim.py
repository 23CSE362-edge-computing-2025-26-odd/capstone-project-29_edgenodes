import pandas as pd
import numpy as np
import os
import time
import random

# Path to traffic.csv
CSV_PATH = os.path.join(r'C:\Users\dudde\2025-09-06-16-37-48', 'traffic.csv')

class TrafficSimulator:
    def __init__(self):
        # Edge is fast (near the traffic light controller)
        self.edge_latency = 0.1    # ~100 ms
        # Cloud is slower (due to network + processing)
        self.cloud_latency = 1.0   # ~1 sec
        self.backup_interval = 5   # Send to cloud every 5 samples

    def derive_metrics(self, row):
        queue = row.get('queue', np.nan)
        density = row.get('density', np.nan)
        occupancy = row.get('occupancy', np.nan)
        speed = row.get('spd', np.nan)

        lane_length = 100  # adjust per SUMO config

        if pd.isna(queue):
            queue = row.get('tl_state', '').count('r') or 0
        if pd.isna(density):
            vehicle_count = row.get('tl_lanes_controlled', '').count(',') + 1 if row.get('tl_lanes_controlled') else 1
            density = vehicle_count / lane_length if lane_length > 0 else 0
        if pd.isna(occupancy):
            occupancy = density * 5 / lane_length if lane_length > 0 else 0
        if pd.isna(speed):
            speed = row.get('speed', 0) or 0

        return queue, density, occupancy, speed

    def predict_duration(self, queue, density, occupancy, speed):
        score = 0.6 * queue + 2.0 * density + 0.1 * occupancy - 0.3 * (speed / 3.6)

        if score > 15:
            return 40
        elif score > 10:
            return 30
        elif score > 5:
            return 20
        else:
            return 10

    def run_simulation(self):
        try:
            # Stream rows one by one (live feed simulation)
            for idx, row in enumerate(pd.read_csv(CSV_PATH, chunksize=1)):
                row = row.iloc[0]  # extract row
                # simulate edge jitter (80–120 ms instead of fixed 100 ms)
                time.sleep(self.edge_latency + random.uniform(-0.02, 0.02))

                # Edge decision
                queue, density, occupancy, speed = self.derive_metrics(row)
                duration = self.predict_duration(queue, density, occupancy, speed)
                timestamp = row.get('dateandtime', f"Time_{idx}")
                print(f"[{timestamp}] Edge: Green light {duration}s "
                      f"(queue={queue:.2f}, density={density:.2f}, "
                      f"occupancy={occupancy:.2f}, speed={speed:.2f})")

                # Cloud backup every Nth row
                if idx % self.backup_interval == 0:
                    # simulate cloud jitter (0.8–1.2s)
                    time.sleep(self.cloud_latency + random.uniform(-0.2, 0.2))
                    print(f"[{timestamp}] Cloud: Backup received "
                          f"(duration={duration}s, q={queue:.2f}, d={density:.2f}, "
                          f"o={occupancy:.2f}, s={speed:.2f})")

        except FileNotFoundError:
            print(f"Error: {CSV_PATH} not found.")

if __name__ == "__main__":
    simulator = TrafficSimulator()
    simulator.run_simulation()
