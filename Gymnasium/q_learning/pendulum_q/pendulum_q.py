import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

divisions = 15

cos_space = np.linspace(-1, 1, divisions)
sin_space = np.linspace(-1, 1, divisions)
vel_space = np.linspace(-8, 8, divisions)
action_space = np.linspace(-2, 2, divisions)


def parse_state(state):
    cos = np.digitize(state[0], cos_space)
    sin = np.digitize(state[1], sin_space)
    vel = np.digitize(state[2], vel_space)
    return cos + sin * (divisions + 1) + vel * (divisions + 1) ** 2


def train(num_episodes):
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01

    q_table = np.zeros(((divisions + 1) ** 3, divisions))

    env = gym.make("Pendulum-v1")

    def choose_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(0, divisions)
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

    plt.savefig(os.path.join(current_dir, "pendulum_q.png"))

    with open(os.path.join(current_dir, "pendulum_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    with open(os.path.join(current_dir, "pendulum_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    env = gym.make("Pendulum-v1", render_mode="human")

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


train(10000)
test(10)
