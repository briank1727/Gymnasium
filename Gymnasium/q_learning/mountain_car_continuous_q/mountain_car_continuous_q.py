import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

divisions = 15

pos_space = np.linspace(-1.2, 0.6, divisions)
vel_space = np.linspace(-0.07, 0.07, divisions)
action_space = np.linspace(-1, 1, divisions)


def parse_state(state):
    pos = np.digitize(state[0], pos_space)
    vel = np.digitize(state[1], vel_space)
    return pos + vel * (divisions + 1)


def train(num_episodes):
    learning_rate = 0.9
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99995
    min_epsilon = 0.01

    q_table = np.zeros(((divisions + 1) ** 3, divisions))

    env = gym.make("MountainCarContinuous-v0")

    def choose_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(0, divisions)
        else:
            return np.argmax(q_table[state, :])

    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        if episode % 100 == 0 and episode > 0:
            print(
                f"On episode {episode}. Average reward: {np.mean(reward_history[episode - 100:episode])}"
            )
        state = env.reset()[0]
        state = parse_state(state)
        terminated = False
        truncated = False
        reward_sum = 0

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(
                [action_space[action]]
            )
            new_state = parse_state(new_state)
            old_reward = q_table[state, action]
            new_max = np.max(q_table[new_state, :])
            q_table[state, action] = (
                1 - learning_rate
            ) * old_reward + learning_rate * (discount_factor * new_max + reward)
            state = new_state
            reward_sum += reward

        reward_history.append(reward_sum)
        epsilon_history.append(epsilon)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(reward_history[max(0, t - 100) : (t + 1)]) / 100

    plt.subplot(121)
    plt.scatter(np.arange(num_episodes), sum_rewards, s=1)

    plt.subplot(122)
    plt.plot(epsilon_history)

    plt.savefig(os.path.join(current_dir, "mountain_car_continuous_q.png"))

    with open(os.path.join(current_dir, "mountain_car_continuous_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    with open(os.path.join(current_dir, "mountain_car_continuous_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    def choose_action(state):
        return np.argmax(q_table[state, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        state = parse_state(state)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(
                [action_space[action]]
            )
            new_state = parse_state(new_state)
            state = new_state

    env.close()


train(100000)
test(10)
