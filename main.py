# main.py
from simulator import TrafficSimulator

CSV_PATH = r"C:\Users\daggu\Downloads\traffic.csv"  

if __name__ == "__main__":
    sim = TrafficSimulator(CSV_PATH)
    sim.run_simulation()
