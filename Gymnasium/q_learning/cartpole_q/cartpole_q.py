import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

pos_space = np.linspace(-2.4, 2.4, 10)
pos_vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-0.2095, 0.2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)


def state_to_vals(state):
    pos = np.digitize(state[0], pos_space)
    pos_vel = np.digitize(state[1], pos_vel_space)
    ang = np.digitize(state[2], ang_space)
    ang_vel = np.digitize(state[3], ang_vel_space)
    return pos, pos_vel, ang, ang_vel


def train(num_episodes):
    env = gym.make("CartPole-v1")

    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1
    epsilon_decay = 0.99995
    min_epsilon = 0.01

    q_table = np.zeros((11, 11, 11, 11, 2))

    reward_history = []
    epsilon_history = []

    def choose_action(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            pos, pos_vel, ang, ang_vel = state_to_vals(state)
            return np.argmax(q_table[pos, pos_vel, ang, ang_vel, :])

    for episode in range(num_episodes):
        if episode % 500 == 0:
            print(f"On episode {episode}")

        state = env.reset()[0]
        terminated = False
        truncated = False
        reward_sum = 0

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, new_reward, terminated, truncated, _ = env.step(action)
            pos, pos_vel, ang, ang_vel = state_to_vals(state)
            old_reward = q_table[pos, pos_vel, ang, ang_vel, action]
            new_pos, new_pos_vel, new_ang, new_ang_vel = state_to_vals(new_state)
            new_max = np.max(q_table[new_pos, new_pos_vel, new_ang, new_ang_vel, :])
            q_table[pos, pos_vel, ang, ang_vel, action] = (
                1 - learning_rate
            ) * old_reward + learning_rate * (discount_factor * new_max + new_reward)
            state = new_state
            reward_sum += new_reward

        reward_history.append(reward_sum)
        epsilon_history.append(epsilon)

        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    env.close()

    sum_rewards = np.zeros(num_episodes - 100)
    for t in range(num_episodes - 100):
        sum_rewards[t] = np.sum(reward_history[t : t + 100]) / 100

    plt.subplot(121)
    plt.plot(sum_rewards)

    plt.subplot(122)
    plt.plot(epsilon_history)

    plt.savefig(os.path.join(current_dir, "cartpole_q.png"))

    with open(os.path.join(current_dir, "cartpole_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    env = gym.make("CartPole-v1", render_mode="human")

    with open(os.path.join(current_dir, "cartpole_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    def choose_action(state):
        pos, pos_vel, ang, ang_vel = state_to_vals(state)
        return np.argmax(q_table[pos, pos_vel, ang, ang_vel, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, new_reward, terminated, truncated, _ = env.step(action)
            state = new_state

    env.close()


# train(100000)
test(10)
