import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

divisions = 10

cos_1_space = np.linspace(-1, 1, divisions)
sin_1_space = np.linspace(-1, 1, divisions)
cos_2_space = np.linspace(-1, 1, divisions)
sin_2_space = np.linspace(-1, 1, divisions)
ang_vel_1_space = np.linspace(-12.6, 12.6, divisions)
ang_vel_2_space = np.linspace(-28.3, 1, divisions)


def parse_state(state):
    cos_1 = np.digitize(state[0], cos_1_space)
    sin_1 = np.digitize(state[1], sin_1_space)
    cos_2 = np.digitize(state[2], cos_2_space)
    sin_2 = np.digitize(state[3], sin_2_space)
    ang_vel_1 = np.digitize(state[4], ang_vel_1_space)
    ang_vel_2 = np.digitize(state[5], ang_vel_2_space)
    return (
        cos_1
        + sin_1 * (divisions + 1)
        + cos_2 * (divisions + 1) ** 2
        + sin_2 * (divisions + 1) ** 3
        + ang_vel_1 * (divisions + 1) ** 4
        + ang_vel_2 * (divisions + 1) ** 5
    )


def train(num_episodes):
    learning_rate = 0.05
    discount_factor = 0.995
    epsilon = 1.0
    epsilon_decay = 0.9999
    min_epsilon = 0.01

    q_table = np.zeros(((divisions + 1) ** 6, 3))

    env = gym.make("Acrobot-v1")

    def choose_action(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state, :])

    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        if episode % 500 == 0:
            print(f"On episode {episode}")
        state = env.reset()[0]
        state = parse_state(state)
        terminated = False
        truncated = False
        reward_sum = 0

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
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

    plt.savefig(os.path.join(current_dir, "acrobot_q.png"))

    with open(os.path.join(current_dir, "acrobot_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    with open(os.path.join(current_dir, "acrobot_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    env = gym.make("Acrobot-v1", render_mode="human")

    def choose_action(state):
        return np.argmax(q_table[state, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        state = parse_state(state)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = parse_state(new_state)
            state = new_state

    env.close()


train(20000)
test(10)
