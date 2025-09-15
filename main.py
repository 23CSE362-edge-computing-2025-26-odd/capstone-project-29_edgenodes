import pandas as pd
from fuzzy_module import compute_green_time
from q_learning import QLearningAgent

MIN_GREEN = 7
MAX_GREEN = 100

# Load SUMO output
df = pd.read_csv('Input/sumo_output.csv')

# Aggregate per intersection (edge) and time_bin
agg_df = df.groupby(['dateandtime', 'edge']).agg(
    vehicle_count=('vehid', 'count'),
    queue_length=('vehid', lambda x: (df.loc[x.index, 'spd']==0).sum()),
    avg_speed=('spd', 'mean')
).reset_index()

# Compute congestion score as queue_length / vehicle_count
agg_df['congestion_score'] = agg_df['queue_length'] / agg_df['vehicle_count']

# Q-learning action space: delta adjustments
actions = list(range(-10, 11, 2))
agent = QLearningAgent(actions, min_green=MIN_GREEN, max_green=MAX_GREEN)

output_rows = []
prev_state = (0, 0)
prev_action = 0

for _, row in agg_df.iterrows():
    intersection_id = row['edge']
    time_bin = row['dateandtime']
    vehicle_count = row['vehicle_count']
    avg_speed = row['avg_speed']
    queue = row['queue_length']
    congestion_score = row['congestion_score']

    # Fuzzy base green
    current_green = compute_green_time(avg_speed, queue, min_bound=MIN_GREEN, max_bound=MAX_GREEN)

    # Q-learning state
    state = (int(avg_speed//10), int(queue//5))

    # Q-learning delta adjustment
    fuzzy_delta = agent.choose_action(state)
    suggested_green_fuzzy = current_green + fuzzy_delta
    suggested_green_fuzzy = max(MIN_GREEN, min(MAX_GREEN, suggested_green_fuzzy))

    # Improved reward
    queue_penalty = - (queue ** 2)
    change_penalty = - abs(fuzzy_delta) * 0.5
    speed_reward = avg_speed / 10
    reward = queue_penalty + speed_reward + change_penalty
    agent.learn(prev_state, prev_action, reward, state)

    output_rows.append({
        'intersection_id': intersection_id,
        'time_bin': time_bin,
        'vehicle_count': vehicle_count,
        'avg_speed': round(avg_speed, 2),
        'congestion_score': round(congestion_score, 2),
        'current_green': round(current_green, 2),
        'fuzzy_delta': fuzzy_delta,
        'suggested_green_fuzzy': round(suggested_green_fuzzy, 2)
    })

    prev_state = state
    prev_action = fuzzy_delta

# Save final CI output
output_df = pd.DataFrame(output_rows)
output_df.to_csv('Output/ci_output.csv', index=False)
print("ci_output.csv generated successfully with all required columns.")
