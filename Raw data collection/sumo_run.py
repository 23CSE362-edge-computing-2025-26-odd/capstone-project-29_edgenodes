import traci
import time
import traci.constants as tc
import pytz
import datetime
import pandas as pd

# Function to get current datetime in Singapore timezone
def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

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

# Function to get lane metrics
def get_lane_metrics(lane):
    queue = traci.lane.getLastStepHaltingNumber(lane)
    veh_count = traci.lane.getLastStepVehicleNumber(lane)
    lane_length = traci.lane.getLength(lane)
    density = veh_count / lane_length if lane_length > 0 else 0
    occupancy = traci.lane.getLastStepOccupancy(lane)
    avg_speed = traci.lane.getLastStepMeanSpeed(lane)
    return queue, density, occupancy, avg_speed

# Decide phase duration based on congestion
def decide_phase_duration(lanes):
    total_queue = 0
    total_density = 0
    total_occupancy = 0
    total_speed = 0
    n = len(lanes)
    for lane in lanes:
        q, d, o, s = get_lane_metrics(lane)
        total_queue += q
        total_density += d
        total_occupancy += o
        total_speed += s

    avg_density = total_density / n if n > 0 else 0
    avg_occupancy = total_occupancy / n if n > 0 else 0
    avg_speed = total_speed / n if n > 0 else 0

    if total_queue > 15 or avg_density > 0.3 or avg_occupancy > 0.3 or avg_speed < 3:
        return 40
    elif total_queue > 10 or avg_density > 0.2 or avg_occupancy > 0.2 or avg_speed < 5:
        return 30
    elif total_queue > 5 or avg_density > 0.1 or avg_occupancy > 0.1:
        return 20
    else:
        return 10

# SUMO command
sumoCmd = ["sumo-gui", "-c", "osm.sumocfg"]
traci.start(sumoCmd)

# Your traffic light ID
MY_TLS = "cluster_1599226662_1599226663_237566456_237567290_#8more"

# Data storage
packBigData = []

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    trafficlights = traci.trafficlight.getIDList()

    for vehid in vehicles:
        # Vehicle data
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

        print(f"Vehicle: {vehid} at datetime: {getdatetime()}")
        print(f"{vehid} >>> Position: {coord} | GPS Position: {gpscoord} | "
              f"Speed: {spd} km/h | EdgeID: {edge} | LaneID: {lane} | "
              f"Distance: {displacement} m | Orientation: {turnAngle} deg | "
              f"Upcoming traffic lights: {nextTLS}")

    # --- Traffic light control ---
    if MY_TLS in trafficlights:
        controlled_lanes = traci.trafficlight.getControlledLanes(MY_TLS)
        new_duration = decide_phase_duration(controlled_lanes)
        traci.trafficlight.setPhaseDuration(MY_TLS, new_duration)
        tl_state = traci.trafficlight.getRedYellowGreenState(MY_TLS)
        tl_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(MY_TLS)
        tl_next_switch = traci.trafficlight.getNextSwitch(MY_TLS)

        print(f"{MY_TLS} -> Adaptive control applied | New duration: {new_duration} | State: {tl_state}")

        tlsList = [MY_TLS, tl_state, new_duration, controlled_lanes, tl_program, tl_next_switch]
        packBigDataLine = flatten_list([vehList, tlsList])
        packBigData.append(packBigDataLine)
    else:
        print(f"ERROR: TLS {MY_TLS} not found in this network!")

    # Vehicle example control
    NEWSPEED = 15  # m/s
    if 'veh2' in vehicles:
        traci.vehicle.setSpeedMode('veh2', 0)
        traci.vehicle.setSpeed('veh2', NEWSPEED)

# End of simulation
traci.close()

# Export to Excel
columnnames = ['dateandtime', 'vehid', 'coord', 'gpscoord', 'spd',
               'edge', 'lane', 'displacement', 'turnAngle', 'nextTLS',
               'tflight', 'tl_state', 'tl_phase_duration',
               'tl_lanes_controlled', 'tl_program', 'tl_next_switch']

dataset = pd.DataFrame(packBigData, columns=columnnames)
dataset.to_csv("output.csv", index=False)
time.sleep(5)
