import traci
import time
import traci.constants as tc
import pytz
import datetime
import pandas as pd

# -------------------- Helper Functions --------------------
# Get current datetime in Singapore timezone
def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    return currentDT.strftime("%Y-%m-%d %H:%M:%S")

# Flatten 2D list into 1D list
def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if isinstance(element, list):
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

# Get lane metrics
def get_lane_metrics(lane):
    queue = traci.lane.getLastStepHaltingNumber(lane)
    veh_count = traci.lane.getLastStepVehicleNumber(lane)
    lane_length = traci.lane.getLength(lane)
    density = veh_count / lane_length if lane_length > 0 else 0
    occupancy = traci.lane.getLastStepOccupancy(lane)
    avg_speed = traci.lane.getLastStepMeanSpeed(lane)
    return queue, density, occupancy, avg_speed

# -------------------- ML Placeholder Model --------------------
def ml_predict_phase_duration(vehicle_metrics):
    """
    Simulated ML model predicting green-light duration.
    vehicle_metrics: list of tuples (queue, density, occupancy, avg_speed)
    """
    total_queue = sum([m[0] for m in vehicle_metrics])
    avg_density = sum([m[1] for m in vehicle_metrics]) / len(vehicle_metrics)
    avg_speed = sum([m[3] for m in vehicle_metrics]) / len(vehicle_metrics)

    # Simple ML-like scoring
    score = 0.6 * total_queue + 2.0 * avg_density - 0.3 * avg_speed

    # Map score to green light duration
    if score > 15:
        return 40
    elif score > 10:
        return 30
    elif score > 5:
        return 20
    else:
        return 10

# -------------------- SUMO Setup --------------------
sumoCmd = ["sumo", "-c", "osm.sumocfg"]
traci.start(sumoCmd)

# Traffic light to control
MY_TLS = "cluster_1599226662_1599226663_237566456_237567290_#8more"

# Data storage
packBigData = []

# -------------------- Simulation Loop --------------------
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    trafficlights = traci.trafficlight.getIDList()

    # --- Collect vehicle data ---
    for vehid in vehicles:
        x, y = traci.vehicle.getPosition(vehid)
        coord = [x, y]
        lon, lat = traci.simulation.convertGeo(x, y)
        gpscoord = [lon, lat]
        spd = round(traci.vehicle.getSpeed(vehid) * 3.6, 2)  # km/h
        edge = traci.vehicle.getRoadID(vehid)
        lane = traci.vehicle.getLaneID(vehid)
        displacement = round(traci.vehicle.getDistance(vehid), 2)
        turnAngle = round(traci.vehicle.getAngle(vehid), 2)
        nextTLS = traci.vehicle.getNextTLS(vehid)

        vehList = [getdatetime(), vehid, coord, gpscoord, spd, edge, lane, displacement, turnAngle, nextTLS]

    # --- Traffic light control with ML ---
    if MY_TLS in trafficlights:
        controlled_lanes = traci.trafficlight.getControlledLanes(MY_TLS)
        vehicle_metrics = [get_lane_metrics(lane) for lane in controlled_lanes]

        # Predict green-light duration using ML placeholder
        new_duration = ml_predict_phase_duration(vehicle_metrics)
        traci.trafficlight.setPhaseDuration(MY_TLS, new_duration)

        tl_state = traci.trafficlight.getRedYellowGreenState(MY_TLS)
        tl_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(MY_TLS)
        tl_next_switch = traci.trafficlight.getNextSwitch(MY_TLS)

        print(f"{MY_TLS} -> ML-based control applied | Duration: {new_duration} | State: {tl_state}")

        tlsList = [MY_TLS, tl_state, new_duration, controlled_lanes, tl_program, tl_next_switch]
        packBigDataLine = flatten_list([vehList, tlsList])
        packBigData.append(packBigDataLine)
    else:
        print(f"ERROR: TLS {MY_TLS} not found in this network!")

    # Example manual vehicle control
    NEWSPEED = 15  # m/s
    if 'veh2' in vehicles:
        traci.vehicle.setSpeedMode('veh2', 0)
        traci.vehicle.setSpeed('veh2', NEWSPEED)

# -------------------- End of Simulation --------------------
traci.close()

# Export collected data to Excel
columnnames = ['dateandtime', 'vehid', 'coord', 'gpscoord', 'spd',
               'edge', 'lane', 'displacement', 'turnAngle', 'nextTLS',
               'tflight', 'tl_state', 'tl_phase_duration',
               'tl_lanes_controlled', 'tl_program', 'tl_next_switch']

dataset = pd.DataFrame(packBigData, columns=columnnames)
dataset.to_csv("traffic.csv", index=False)
time.sleep(5)