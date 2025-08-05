import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))


def print_q_table(q_table):
    print()
    for observation in range(len(q_table)):
        print(
            f"{observation}, {",".join([f"{num:0.2f}" for num in q_table[observation]])}"
        )


def train(num_episodes, is_slippery):
    env = gym.make("CliffWalking-v1", is_slippery=is_slippery)
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state, :])

    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"On episode {episode}")

        state = env.reset()[0]
        terminated = False
        truncated = False
        reward_sum = 0

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            old_value = q_table[state, action]
            new_max = np.max(q_table[new_state, :])
            q_table[state, action] = (1 - learning_rate) * old_value + learning_rate * (
                reward + discount_factor * new_max
            )
            state = new_state
            reward_sum += reward

        reward_history.append(reward_sum)
        epsilon_history.append(epsilon)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()

    sum_rewards = np.zeros(num_episodes - 100)
    for t in range(num_episodes - 100):
        sum_rewards[t] = np.sum(reward_history[t : t + 100]) / 100

    plt.subplot(121)
    plt.plot(sum_rewards)

    plt.subplot(122)
    plt.plot(epsilon_history)

    plt.savefig(os.path.join(current_dir, "cliff_walking_q.png"))

    with open(os.path.join(current_dir, "cliff_walking_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes, is_slippery):
    env = gym.make("CliffWalking-v1", render_mode="human", is_slippery=is_slippery)

    with open(os.path.join(current_dir, "cliff_walking_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    print_q_table(q_table)

    def choose_action(state):
        return np.argmax(q_table[state, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

    env.close()


is_slippery = True
# train(10000, is_slippery)
test(10, is_slippery)
