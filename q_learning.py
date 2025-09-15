import numpy as np
import random

class QLearningAgent:
    def __init__(self,actions,min_green=5,max_green=120,alpha=0.1,gamma=0.9,epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_green = min_green
        self.max_green = max_green

    def get_q(self,state,action):
        return self.q_table.get((state,action),0.0)

    def choose_action(self,state):
        if random.uniform(0,1) < self.epsilon:
            green_time = random.choice(self.actions)
        else:
            q_values = [self.get_q(state,a) for a in self.actions]
            max_q = max(q_values)
            candidates = [a for a,q in zip(self.actions,q_values) if q == max_q]
            green_time = random.choice(candidates)
        # Clamp dynamically
        if green_time < self.min_green:
            green_time = self.min_green
        elif green_time > self.max_green:
            green_time = self.max_green
        return green_time

    def learn(self,state,action,reward,next_state):
        old_q = self.get_q(state,action)
        future_q = max([self.get_q(next_state,a) for a in self.actions])
        self.q_table[(state,action)] = old_q + self.alpha*(reward + self.gamma*future_q - old_q)
