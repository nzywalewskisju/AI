import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------------------
# Core RL pieces (no pygame)
# -------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class CarEnv:
    """
    Same environment as the visualizer, but without pygame.
    """

    def __init__(self):
        self.GRID_SIZE = 10
        self.FIELD_WIDTH = self.GRID_SIZE
        self.FIELD_HEIGHT = self.GRID_SIZE

        # Actions: 0 Forward, 1 Right, 2 Left, 3 Stay
        self.ACTION_SIZE = 4
        self.ACTION_MAP = {
            0: "Forward (F)",
            1: "Turn Right (R)",
            2: "Turn Left (L)",
            3: "Stay (S)",
        }

        # State: [x, y, orientation]
        self.OBSERVATION_SIZE = 3

        self.COURSE = [
            "S.........",
            ".XX.X.X...",
            "..X.X.X.X.",
            ".X.X.G..X.",
            ".X.X.XXX.X",
            ".X....X..X",
            ".XXXX.X.X.",
            "....X.X.X.",
            ".X.X......",
            "..........",
        ]

        self.GOAL_POS = self._find_char_pos("G")
        self.START_POS = self._find_char_pos("S")

    def _find_char_pos(self, char):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.COURSE[y][x] == char:
                    return [x, y]
        return None

    def reset(self):
        self.car_pos = self.START_POS.copy()
        self.car_orientation = 1  # start facing East
        return self._get_observation()

    def _get_observation(self):
        return np.array(self.car_pos + [self.car_orientation], dtype=np.float32)

    def step(self, action):
        old_dist = math.dist(self.car_pos, self.GOAL_POS)

        reward = -0.1
        done = False

        current_x, current_y = self.car_pos
        current_o = self.car_orientation
        next_pos = self.car_pos.copy()
        next_o = self.car_orientation

        info = {"action": self.ACTION_MAP[action], "outcome": "Moved"}

        if action == 0:
            # forward
            dx, dy = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[current_o]
            next_pos[0] += dx
            next_pos[1] += dy
            info["outcome"] = "Forward"
        elif action == 1:
            # right
            next_o = (current_o + 1) % 4
            info["outcome"] = "Turn Right"
        elif action == 2:
            # left
            next_o = (current_o - 1) % 4
            info["outcome"] = "Turn Left"
        elif action == 3:
            # stay
            info["outcome"] = "Stay"

        new_x, new_y = next_pos

        # wall
        if not (0 <= new_x < self.GRID_SIZE and 0 <= new_y < self.GRID_SIZE):
            reward = -10.0
            done = True
            info["outcome"] = "CRASHED (Wall)"
        else:
            # obstacle
            if self.COURSE[new_y][new_x] == "X":
                reward = -10.0
                done = True
                info["outcome"] = "CRASHED (Obstacle)"
            else:
                self.car_pos = next_pos
                self.car_orientation = next_o

                new_dist = math.dist(self.car_pos, self.GOAL_POS)
                reward += (old_dist - new_dist) * 0.5

                if self.car_pos == self.GOAL_POS:
                    reward = 50.0
                    info["outcome"] = "REACHED GOAL"
                    done = True

        return self._get_observation(), reward, done, info


class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, env, gamma, buffer_capacity, learning_rate):
        self.state_size = env.OBSERVATION_SIZE
        self.action_size = env.ACTION_SIZE
        self.gamma = gamma

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_capacity)

        self.epsilon = 0.0
        self.steps_done = 0

    def act(self, state, eps_threshold):
        self.epsilon = eps_threshold
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.model(state_batch).gather(
            1, action_batch.unsqueeze(1)
        ).squeeze()

        next_action_selection = self.model(next_state_batch).max(1)[1].detach().unsqueeze(1)
        next_state_values = self.target_model(next_state_batch).gather(
            1, next_action_selection
        ).squeeze().detach()
        next_state_values[done_batch] = 0.0

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


# -------------------------------------------------------------------
# Gamma experiments for Part 2
# -------------------------------------------------------------------

BUFFER_CAPACITY = 10000
LEARNING_RATE = 0.0005
BATCH_SIZE = 64

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000 # winning decay from part 1

NUM_EPISODES = 1000
TARGET_UPDATE_FREQ = 500
MAX_STEPS = 200


def train_for_gamma(gamma, label):
    env = CarEnv()
    agent = DQNAgent(env, gamma, BUFFER_CAPACITY, LEARNING_RATE)

    successes = 0
    episode_rewards = []
    steps_to_goal = []          # store path length for successful episodes

    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                -agent.steps_done / float(EPS_DECAY)
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

        # reached goal
        if env.car_pos == env.GOAL_POS:
            successes += 1
            steps_to_goal.append(steps)

        episode_rewards.append(total_reward)
        print(
            f"{label} gamma={gamma:.2f} | Episode {ep + 1} | "
            f"Reward {total_reward:.2f} | Steps {steps}"
        )

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    success_rate = successes / NUM_EPISODES

    if steps_to_goal:
        avg_steps_to_goal = sum(steps_to_goal) / len(steps_to_goal)
        best_steps = min(steps_to_goal)
    else:
        avg_steps_to_goal = None
        best_steps = None

    print(f"\n[{label}] gamma={gamma:.2f}")
    print(f"Average reward over {NUM_EPISODES} episodes: {avg_reward:.2f}")
    print(
        f"Goals reached: {successes}/{NUM_EPISODES} "
        f"({success_rate * 100:.1f} percent)"
    )
    print(f"Average steps to goal (successful episodes only): {avg_steps_to_goal}")
    print(f"Shortest path found: {best_steps} steps\n")

    # return metrics so main can build a table
    return {
        "label": label,
        "gamma": gamma,
        "successes": successes,
        "episodes": NUM_EPISODES,
        "success_rate": success_rate,
        "avg_steps": avg_steps_to_goal,
        "shortest_path": best_steps,
    }


def main():
    gamma_settings = {
        "Low": 0.30,     # short sighted
        "Medium": 0.60,  # balanced
        "High": 0.99,    # far sighted
    }

    results = []

    for label, gamma in gamma_settings.items():
        print(f"\n=== Starting {label} Gamma run (gamma = {gamma}) ===")
        metrics = train_for_gamma(gamma, label)
        results.append(metrics)

    # print summary table
    print("\nSummary across gamma values\n")

    headers = [
        "Case",
        "Gamma",
        "Successes",
        "Success rate",
        "Avg steps to goal",
        "Shortest path",
    ]

    # simple formatting
    row_format = "{:<8} {:<7} {:<10} {:<13} {:<18} {:<14}"

    print(row_format.format(*headers))
    print("-" * 70)

    for m in results:
        if m["avg_steps"] is None:
            avg_steps_str = "n/a"
            shortest_str = "n/a"
        else:
            avg_steps_str = f"{m['avg_steps']:.1f}"
            shortest_str = str(m["shortest_path"])

        print(
            row_format.format(
                m["label"],
                f"{m['gamma']:.2f}",
                f"{m['successes']}/{m['episodes']}",
                f"{m['success_rate'] * 100:.1f}%",
                avg_steps_str,
                shortest_str,
            )
        )


if __name__ == "__main__":
    main()