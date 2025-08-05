import gymnasium as gym
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))


def train(num_episodes):
    env = gym.make("MountainCar-v0")

    pos_space = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], 20
    )
    vel_space = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], 20
    )

    q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01

    def choose_action(pos, vel):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[pos, vel, :])

    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print(f"On episode {episode}")
        state = env.reset()[0]
        pos = np.digitize(state[0], pos_space)
        vel = np.digitize(state[1], vel_space)
        terminated = False
        truncated = False
        reward_sum = 0
        while not terminated and not truncated:
            action = choose_action(pos, vel)
            new_state, new_reward, terminated, truncated, _ = env.step(action)
            new_pos = np.digitize(new_state[0], pos_space)
            new_vel = np.digitize(new_state[1], vel_space)
            old_reward = q_table[pos, vel, action]
            new_max = np.max(q_table[new_pos, new_vel, :])
            q_table[pos, vel, action] = (
                1 - learning_rate
            ) * old_reward + learning_rate * (new_reward + discount_factor * new_max)
            reward_sum += new_reward
            pos = new_pos
            vel = new_vel

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

    plt.savefig(os.path.join(current_dir, "mountain_car_q.png"))

    with open(os.path.join(current_dir, "mountain_car_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    env = gym.make("MountainCar-v0", render_mode="human")

    pos_space = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], 20
    )
    vel_space = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], 20
    )

    with open(os.path.join(current_dir, "mountain_car_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    def choose_action(pos, vel):
        return np.argmax(q_table[pos, vel, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        pos = np.digitize(state[0], pos_space)
        vel = np.digitize(state[1], vel_space)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(pos, vel)
            new_state, reward, terminated, truncated, _ = env.step(action)
            pos = np.digitize(new_state[0], pos_space)
            vel = np.digitize(new_state[1], vel_space)

    env.close()


train(10000)
test(10)
