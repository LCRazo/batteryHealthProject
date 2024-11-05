from battery_env import BatteryEnv  # Assuming the environment is in a file named 'battery_env.py'
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time

num_cell = 5


# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Stores experience and enables sampling for replay in optimal model function
# Finds correlation between experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            np.array(next_state),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, env):
    if random.random() > epsilon:
        # Use policy network to predict Q-values
        with torch.no_grad():
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            action = (q_values > 0).cpu().numpy().astype(int).flatten()  # Convert to binary actions
    else:
        # Exploration: choose a random action for each cell
        action = env.action_space.sample()  # Returns a binary array of length `num_cell`
    return action


# Optimize the model using DQN
def optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = actions.unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    # Compute the target Q-values
    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute the loss and optimize the model
    loss = nn.functional.mse_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    loss.backward()
    optimizer.step()


# Main training loop
def train_dqn():
    env = BatteryEnv()
    initial_state = env.reset()
    print("Initial State:", initial_state)

    input_dim = int(np.prod(env.observation_space.shape))  # Flattened observation space, gives 10
    output_dim = env.action_space.shape[0]  # Should be 5 for MultiBinary action space

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0003)  # Fine-tuned learning rate
    memory = ReplayBuffer(50000)  # Saves trajectory to better training and stabilizes
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    sync_target_steps = 10

    current_time = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/batteryHealth_{current_time}')

    num_episodes = 5000
    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        total_reward = 0
        done = False

        for t in range(500):
            action = select_action(state, policy_net, epsilon, env)  # Choose a random action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Reward shaping: reward for progress towards the goal
            soc, switch_state, *_ = next_state
            reward += abs(soc.mean() - 0.5)

            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            #optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size)

            if done:
                break

        # Adjust epsilon for the next episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update the target network every few episodes
        if episode % sync_target_steps == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log total reward and print progress
        writer.add_scalar('Total_Reward_per_Episode', total_reward, episode)
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # Save the model after training
    torch.save(policy_net.state_dict(), f"dqn_batteryHealth_{current_time}.pth")
    print(f"Model saved as 'dqn_batteryHealth_{current_time}.pth'")

    writer.close()
    env.close()


if __name__ == "__main__":
       train_dqn()


