import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os

BASE_RANDOM_SEED = 58922320


def train_q_learning(
    env_name: str,
    seed: int = BASE_RANDOM_SEED,
    use_action_mask: bool = True,
    num_episodes: int = 10000,
    learning_rate: float = 0.01,
    discount_factor: float = 0.99,
    start_epsilon: float = 1,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.01,
) -> dict:
    print(f"Starting {env_name} with seed {seed} with {num_episodes} episodes!")

    env = gym.make(env_name)
    np.random.seed(seed)
    random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    epsilon = start_epsilon
    reward_history = []
    epsilon_history = []

    def choose_action(state, action_mask=None):
        if np.random.random() < epsilon:
            if use_action_mask:
                valid_actions = np.nonzero(action_mask == 1)[0]
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, n_actions)
        else:
            if use_action_mask:
                valid_actions = np.nonzero(action_mask == 1)[0]
                return valid_actions[np.argmax(q_table[state, valid_actions])]
            else:
                return np.argmax(q_table[state])

    for episode in range(num_episodes):
        state, info = env.reset()
        reward_sum = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action_mask = info["action_mask"] if use_action_mask else None
            action = choose_action(state, action_mask)
            new_state, new_reward, terminated, truncated, info = env.step(action)
            reward_sum += new_reward

            if not terminated and not truncated:
                if use_action_mask:
                    new_action_mask = info["action_mask"]
                    valid_next_actions = np.nonzero(new_action_mask == 1)[0]
                    if len(valid_next_actions) > 0:
                        next_max = np.max(q_table[new_state, valid_next_actions])
                    else:
                        next_max = 0
                else:
                    next_max = np.max(q_table[new_state])

                q_table[state, action] = q_table[state, action] + learning_rate * (
                    new_reward + discount_factor * next_max - q_table[state, action]
                )

            state = new_state

        reward_history.append(reward_sum)
        epsilon_history.append(epsilon)

        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    print(
        f"Finished {env_name} with seed {seed} finished training {num_episodes} episodes!"
    )
    env.close()
    return {
        "reward_history": reward_history,
        "epsilon_history": epsilon_history,
        "q_table": q_table,
    }


num_runs = 12
num_episodes = 5000
learning_rate = 0.01
discount_factor = 0.99
start_epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01

seeds = [BASE_RANDOM_SEED + i for i in range(num_runs)]

masked_results_list = []
unmasked_results_list = []
print()
for i, seed in enumerate(seeds):
    print(f"Run {i + 1}/{num_runs} with seed {seed}")

    masked_results = train_q_learning(
        "Taxi-v3",
        seed=seed,
        use_action_mask=True,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        start_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
    )
    masked_results_list.append(masked_results)

    unmasked_results = train_q_learning(
        "Taxi-v3",
        seed=seed,
        use_action_mask=False,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        start_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
    )
    unmasked_results_list.append(unmasked_results)

savefig_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(savefig_folder, exist_ok=True)

plt.plot(masked_results_list[0]["epsilon_history"])
plt.savefig(
    os.path.join(savefig_folder, "taxi_v3_action_masking_epsilon.png"),
    bbox_inches="tight",
    dpi=150,
)

plt.clf()

plt.figure(figsize=(12, 8), dpi=100)

for i, (masked_results, unmasked_results) in enumerate(
    zip(masked_results_list, unmasked_results_list)
):
    plt.plot(
        masked_results["reward_history"],
        label="With Action Masking" if i == 0 else None,
        color="blue",
        alpha=0.1,
    )
    plt.plot(
        unmasked_results["reward_history"],
        label="Without Action Masking" if i == 0 else None,
        color="red",
        alpha=0.1,
    )

masked_mean_curve = np.mean([r["reward_history"] for r in masked_results_list], axis=0)
unmasked_mean_curve = np.mean(
    [r["reward_history"] for r in unmasked_results_list], axis=0
)

plt.plot(
    masked_mean_curve, label="With Action Masking (Mean)", color="blue", linewidth=2
)
plt.plot(
    unmasked_mean_curve,
    label="Without Action Masking (Mean)",
    color="red",
    linewidth=2,
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance: Q-Learning with vs without Action Masking")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    os.path.join(savefig_folder, "taxi_v3_action_masking_comparison.png"),
    bbox_inches="tight",
    dpi=150,
)

best_q_table = masked_results_list[
    np.argmax([np.max(arr["reward_history"]) for arr in masked_results_list])
]["q_table"]

with open(os.path.join(savefig_folder, "taxi_game_q.pkl"), "wb") as f:
    pickle.dump(best_q_table, f)
