import numpy as np
from RowColArmyGame_3x3 import RowColArmyGame_3x3
import random

env = RowColArmyGame_3x3()

alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.997
min_epsilon = 0.01
num_episodes = 10000
max_steps = 20

q_table = np.zeros((env.observation_space_n, env.action_space_n))


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space_sample()
    else:
        return np.argmax(q_table[state, :])


for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated = env.step(action)

        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * next_max
        )

        state = next_state

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env = RowColArmyGame_3x3()

for episode in range(5):
    state = env.reset()
    print("Episode", episode)
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        print(f"Executing {action}")
        next_state, reward, done, truncated = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print(f"\nFinished with reward {reward}!\n-----------------------------\n")
            break
