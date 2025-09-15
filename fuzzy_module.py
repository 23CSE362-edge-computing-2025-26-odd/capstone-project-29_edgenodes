import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


MIN_GREEN = 5
MAX_GREEN = 120

speed = ctrl.Antecedent(np.arange(0,101,1),'speed')
queue_length = ctrl.Antecedent(np.arange(0,51,1),'queue_length')
green_time = ctrl.Consequent(np.arange(0,MAX_GREEN+1,1),'green_time')

speed['low'] = fuzz.trimf(speed.universe,[0,0,50])
speed['medium'] = fuzz.trimf(speed.universe,[20,50,80])
speed['high'] = fuzz.trimf(speed.universe,[50,100,100])

queue_length['short'] = fuzz.trimf(queue_length.universe,[0,0,20])
queue_length['medium'] = fuzz.trimf(queue_length.universe,[10,25,40])
queue_length['long'] = fuzz.trimf(queue_length.universe,[30,50,50])

green_time['short'] = fuzz.trimf(green_time.universe,[0,0,MAX_GREEN*0.33])
green_time['medium'] = fuzz.trimf(green_time.universe,[MAX_GREEN*0.2,MAX_GREEN*0.5,MAX_GREEN*0.8])
green_time['long'] = fuzz.trimf(green_time.universe,[MAX_GREEN*0.6,MAX_GREEN,MAX_GREEN])

rules = [
    ctrl.Rule(speed['low'] & queue_length['long'],green_time['long']),
    ctrl.Rule(speed['low'] & queue_length['medium'],green_time['medium']),
    ctrl.Rule(speed['low'] & queue_length['short'],green_time['medium']),
    ctrl.Rule(speed['medium'] & queue_length['long'],green_time['long']),
    ctrl.Rule(speed['medium'] & queue_length['medium'],green_time['medium']),
    ctrl.Rule(speed['medium'] & queue_length['short'],green_time['short']),
    ctrl.Rule(speed['high'] & queue_length['long'],green_time['medium']),
    ctrl.Rule(speed['high'] & queue_length['medium'],green_time['short']),
    ctrl.Rule(speed['high'] & queue_length['short'],green_time['short'])
]

green_ctrl = ctrl.ControlSystem(rules)
green_simulator = ctrl.ControlSystemSimulation(green_ctrl)

def compute_green_time(speed_val,queue_val,min_bound=MIN_GREEN,max_bound=MAX_GREEN):
    global MIN_GREEN, MAX_GREEN
    MIN_GREEN = min_bound
    MAX_GREEN = max_bound
    green_simulator.input['speed'] = speed_val
    green_simulator.input['queue_length'] = queue_val
    green_simulator.compute()
    green_time = green_simulator.output['green_time']
    if green_time < MIN_GREEN:
        green_time = MIN_GREEN
    elif green_time > MAX_GREEN:
        green_time = MAX_GREEN
    return green_time
