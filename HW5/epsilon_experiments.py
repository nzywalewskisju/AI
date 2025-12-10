import math
import matplotlib.pyplot as plt

# adjust name if your file is called something else
from car_dqn_visualizer import CarEnv, DQNAgent

# Hyperparameters
GAMMA = 0.9
BUFFER_CAPACITY = 10000
LEARNING_RATE = 0.0005
BATCH_SIZE = 64

EPS_START = 1.0
EPS_END = 0.01

NUM_EPISODES = 5000
TARGET_UPDATE_FREQ = 500
MAX_STEPS = 100        # cap per episode so rewards do not explode


def moving_average(values, window):
    """Simple moving average for smoothing reward curves."""
    avgs = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        avgs.append(sum(window_vals) / len(window_vals))
    return avgs


def train_for_decay(eps_decay, label):
    """
    Train a DQN agent with a specific epsilon decay rate.
    Returns list of total reward per episode.
    """
    env = CarEnv()
    agent = DQNAgent(env, GAMMA, BUFFER_CAPACITY, LEARNING_RATE)

    episode_rewards = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                -agent.steps_done / float(eps_decay)
            )

            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > BATCH_SIZE:
                agent.learn(BATCH_SIZE)

            if agent.steps_done > 0 and agent.steps_done % TARGET_UPDATE_FREQ == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        print(f"{label} | Episode {ep + 1} | Reward {total_reward:.2f}")

    return episode_rewards


def main():
    # decay settings for the three cases
    decay_settings = {
        "High": 1000,
        "Medium": 5000,
        "Low": 20000
    }

    raw_results = {}
    smoothed_results = {}
    window = 50

    # run training once for each decay setting
    for label, eps_decay in decay_settings.items():
        print(f"\n=== Starting {label} Decay Run (Decay = {eps_decay}) ===")
        rewards = train_for_decay(eps_decay, label)
        raw_results[label] = rewards
        smoothed_results[label] = moving_average(rewards, window)

    # compute global y-limits so all plots use the same scale
    all_vals = [v for series in smoothed_results.values() for v in series]
    y_min = min(all_vals) - 5
    y_max = max(all_vals) + 5

    episodes = range(1, NUM_EPISODES + 1)

    # --------- Figure 1: combined plot ----------
    plt.figure()
    for label, smoothed in smoothed_results.items():
        plt.plot(episodes, smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward per Episode")
    plt.title("Effect of Epsilon Decay Rate on DQN Performance")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # --------- Figure 2: High decay only ----------
    plt.figure()
    plt.plot(episodes, smoothed_results["High"], label="High", color="tab:blue")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward per Episode")
    plt.title("High Epsilon Decay")
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # --------- Figure 3: Medium decay only ----------
    plt.figure()
    plt.plot(episodes, smoothed_results["Medium"], label="Medium", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward per Episode")
    plt.title("Medium Epsilon Decay")
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # --------- Figure 4: Low decay only ----------
    plt.figure()
    plt.plot(episodes, smoothed_results["Low"], label="Low", color="tab:green")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward per Episode")
    plt.title("Low Epsilon Decay")
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()