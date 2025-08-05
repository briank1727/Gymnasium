import gymnasium as gym
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))


def train(num_episodes):
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.999995
    min_epsilon = 0.01

    env = gym.make("Blackjack-v1")

    q_table = {}
    for player_sum in range(32):
        for dealer_sum in range(11):
            for usable_ace in range(2):
                q_table[(player_sum, dealer_sum, usable_ace)] = np.zeros(2)

    reward_history = []
    epsilon_history = []

    def choose_action(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state])

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"On Episode {episode}")
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            old_value = q_table[state][action]
            next_max = np.max(q_table[new_state])
            q_table[state][action] = (1 - learning_rate) * old_value + learning_rate * (
                reward + discount_factor * next_max
            )
            state = new_state

        reward_history.append(reward)
        epsilon_history.append(epsilon)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(reward_history[max(0, t - 100) : (t + 1)])

    plt.subplot(121)
    plt.scatter(np.arange(num_episodes), sum_rewards, s=1)

    plt.subplot(122)
    plt.plot(epsilon_history)

    plt.savefig(os.path.join(current_dir, "blackjack_q.png"))

    with open(os.path.join(current_dir, "blackjack_q.pkl"), "wb") as f:
        pickle.dump(q_table, f)


def test(num_episodes):
    env = gym.make("Blackjack-v1", render_mode="human")

    with open(os.path.join(current_dir, "blackjack_q.pkl"), "rb") as f:
        q_table = pickle.load(f)

    def choose_action(state):
        return np.argmax(q_table[state])

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

    env.close()


# train(1000000)
test(10)
