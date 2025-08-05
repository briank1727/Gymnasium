import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os


def run(num_episodes, is_training=False, render=True, is_slippery=False):
    env = gym.make(
        "Taxi-v3",
        render_mode="human" if render else None,
    )
    learning_rate = 0.01
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 1 / (num_episodes)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(os.path.join(current_dir, "taxi_game_q.pkl"), "rb") as f:
            q_table = pickle.load(f)

    def choose_action(state):
        if is_training and random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state, :])

    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        reward = 0

        while not terminated and not truncated:
            action = choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            if not is_training:
                env.render()
            old_value = q_table[state, action]

            next_max = np.max(q_table[next_state, :])
            if is_training:
                q_table[state, action] = (
                    1 - learning_rate
                ) * old_value + learning_rate * (reward + discount_factor * next_max)

            state = next_state

        reward_history.append(reward)
        epsilon_history.append(epsilon)
        epsilon = max(0, epsilon - epsilon_decay)
    if not is_training:
        env.render()
    env.close()

    sum_rewards = [
        np.sum(reward_history[max(0, i - 100) : i + 1]) for i in range(num_episodes)
    ]
    plt.subplot(121)
    plt.plot(sum_rewards)

    plt.subplot(122)
    plt.plot(epsilon_history)
    plt.savefig(os.path.join(current_dir, "taxi_game_q.png"))

    if is_training:
        with open(os.path.join(current_dir, "taxi_game_q.pkl"), "wb") as f:
            pickle.dump(q_table, f)


if __name__ == "__main__":
    # run(15000, is_training=True, render=False)
    run(10, is_training=False, render=True)
