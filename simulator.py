
import pandas as pd
import time
import random
from predictor import TrafficPredictor

class TrafficSimulator:
    def __init__(self, csv_path, backup_interval=5):
        self.edge_latency = 0.1    # ~100 ms
        self.cloud_latency = 1.0   # ~1 sec
        self.backup_interval = backup_interval

        try:
            self.data = pd.read_csv(csv_path)
            self.data.columns = self.data.columns.str.strip()
            print(f"Loaded {len(self.data)} rows from CSV")
        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}")
            self.data = pd.DataFrame()

    def derive_metrics(self, row):
        # Normalize columns and fill missing values
        queue = row.get('queue', row.get('Queue', row.get('QUEUE', 0)))
        density = row.get('density', row.get('Density', row.get('DENSITY', 0)))
        occupancy = row.get('occupancy', row.get('Occupancy', row.get('OCCUPANCY', 0)))
        speed = row.get('spd', row.get('speed', row.get('Speed', 0)))

        # Fill NaNs
        queue = 0 if pd.isna(queue) else queue
        density = 0 if pd.isna(density) else density
        occupancy = 0 if pd.isna(occupancy) else occupancy
        speed = 0 if pd.isna(speed) else speed

        return queue, density, occupancy, speed

    def run_simulation(self):
        if self.data.empty:
            print("No data to process. Exiting.")
            return

        for idx, row in self.data.iterrows():
            time.sleep(self.edge_latency + random.uniform(-0.02, 0.02))
            queue, density, occupancy, speed = self.derive_metrics(row)
            duration = TrafficPredictor.predict_duration(queue, density, occupancy, speed)
            timestamp = row.get('dateandtime', row.get('DateTime', row.get('timestamp', f"Time_{idx}")))

            print(f"[{timestamp}] Edge: Green light {duration}s "
                  f"(queue={queue}, density={density}, occupancy={occupancy}, speed={speed})")

            # Cloud backup
            if idx % self.backup_interval == 0:
                time.sleep(self.cloud_latency + random.uniform(-0.2, 0.2))
                print(f"[{timestamp}] Cloud: Backup received - Duration: {duration}s "
                      f"(queue={queue}, density={density}, occupancy={occupancy}, speed={speed})")
