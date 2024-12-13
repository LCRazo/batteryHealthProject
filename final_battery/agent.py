from battery_env import BatteryEnv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque  # For replay buffer
from torch.utils.tensorboard import SummaryWriter  # For logging training metrics
import pandas as pd  # For saving logs to CSV
import os  # For managing file paths
import time  # For timestamps


# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()#256
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)  # Dropout with a 20% probability
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer to store experiences
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Add a new experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        # Return the current number of stored experiences
        return len(self.buffer)

# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, env):

    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)  # Prepare input
            q_values = policy_net(state)
            action = (q_values > 0).cpu().numpy().astype(int).flatten()  # Binary actions
    else:
        action = env.action_space.sample()  # Random action
    return action

# Optimize the model using experiences from the replay buffer
def optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size, writer, global_step):

    if len(memory) < batch_size:
        return

    # Sample a batch of experiences
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).view(batch_size, -1)
    next_states = torch.tensor(next_states, dtype=torch.float32).view(batch_size, -1)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Calculate current Q-values for the taken actions
    current_q_values = (policy_net(states) * actions).sum(dim=1, keepdim=True)

    # Calculate target Q-values using the target network
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1).type(torch.float32)
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Calculate the loss (Mean Squared Error)
    loss = nn.functional.mse_loss(current_q_values, target_q_values)

    #Backpropagate the loss
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
    loss.backward()
    optimizer.step()

    # Log the loss value
    writer.add_scalar('Loss', loss.item(), global_step)


def train_dqn():
    def update_epsilon(current_epsilon, reward_window_local):
        recent_rewards = list(reward_window_local)
        if len(reward_window_local) == reward_plateau_threshold:
            if max(recent_rewards) - min(recent_rewards) < 5:
                current_epsilon = min(current_epsilon * epsilon_increase_factor, 1.0)
        else:
            current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
        return current_epsilon

    env = BatteryEnv()
    input_dim = int(np.prod(env.observation_space.shape))
    output_dim = env.action_space.shape[0]

    # Initialize Q-networks
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer and replay buffer
    #optimizer = optim.Adam(policy_net.parameters(), lr=0.0003, weight_decay=1e-5)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001, weight_decay=1e-5)
    memory = ReplayBuffer(100_000)

    # Exploration parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999
    reward_plateau_threshold = 100
    reward_window = deque(maxlen=reward_plateau_threshold)  # Track recent rewards
    epsilon_increase_factor = 1.2  # Increase epsilon by 20% on plateau
    gamma = 0.99
    batch_size = 64
    sync_target_steps = 100

    # Logging setup
    current_time = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'/Users/sheilaangeles/PycharmProjects/pythonProject5/.venv/runs/batteryHealth_{current_time}')

    # Global step counter for loss logging
    global_step = 0
    # Training loop
    num_episodes = 50_000
    # Logs to save CSV data
    logs = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        total_reward = 0
        soc_values = []  # List to store SOC values for this episode
        temp_values = []  # List to store temperature values for this episode
        switch_frequencies = []  # List to store switching states for this episode

        # Test the reward calculation
        env = BatteryEnv()
        env.reset()

        for t in range(1000):  # 500 time steps per episode
            action = select_action(state, policy_net, epsilon, env)
            next_state, reward, done, _, _ = env.step(action)

            # Get reward breakdown from the environment
            r_soc, r_temp, usable_capacity_reward, r_switch, penalty, r_balance, _, _ = env.calculate_reward()

            logs.append({
                "episode": episode,
                "time_step": t,
                "reward": reward,  # Total reward
                "soc": next_state[:, 0].tolist(),  # SOC values for all cells
                "temperature": next_state[:, 2].tolist(),  # Temperature values for all cells
                "actions": action.tolist(),  # Action taken
                "voltage": env.voltage.tolist(),  # Voltage values
                "r_soc": r_soc,  # SOC Variance Reward
                "r_temp": r_temp,  # Temperature Variance Reward
                "usable_capacity_reward": usable_capacity_reward,  # Usable Capacity Reward
                "r_switch": r_switch,  # Switching Penalty
                "penalty": penalty,  # Extreme Condition Penalty
                "r_balance": r_balance,  # SOC Balance Reward
                #"final_reward": final_reward  # Final reward
            })

            # Append SOC, temperature, and switch states
            soc_values.extend(next_state[:, 0])  # SOC values for all cells
            temp_values.extend(next_state[:, 2])  # Temperature values for all cells
            switch_frequencies.extend(action)  # Switch states for all cells

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Optimize the model and log loss
            optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size, writer, global_step)

            global_step += 1  # Increment global step for logging

            if done:
                break
        # Update epsilon after each episode
        reward_window.append(total_reward)
        epsilon = update_epsilon(epsilon, reward_window)

        # Calculate metrics
        avg_soc_variance = np.var(soc_values)  # SOC variance
        avg_soc = np.mean(soc_values)
        max_temp = np.max(temp_values)  # Maximum temperature
        min_temp = np.min(temp_values)  # Minimum temperature
        avg_switch_frequency = np.mean(switch_frequencies)  # Average switching frequency

        # Log metrics to TensorBoard
        writer.add_scalar('SOC Variance/Average', avg_soc_variance, episode)
        writer.add_scalar('SOC Average', avg_soc, episode)
        writer.add_scalar('Temperature/Max', max_temp, episode)
        writer.add_scalar('Temperature/Min', min_temp, episode)
        writer.add_scalar('Temperature/Mean', np.mean(temp_values), episode)
        writer.add_scalar('Switching Frequency/Average', avg_switch_frequency, episode)
        writer.add_scalar('Total_Reward_per_Episode', total_reward, episode)

        # Sync target network
        if episode % sync_target_steps == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print episode details including SOC values every 500 episodes
        if episode % 10 == 0:
            print(
                f"Episode {episode}, Total Reward: {total_reward:.2f}, Avg Switching Freq: {avg_switch_frequency:.2f}, Epsilon: {epsilon:.4f}")

        # Save model and close the environment
        model_file = f"dqn_batteryHealth_{current_time}.pth"
        torch.save(policy_net.state_dict(), model_file)
        #print(f"Model saved to {model_file}")

    # Save logs to a CSV file after training
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"battery_dqn_logs_{current_time}.csv")
    pd.DataFrame(logs).to_csv(log_file, index=False)
    print(f"Logs saved to {log_file}")


    # Save model after training
    model_file = f"dqn_batteryHealth_{current_time}.pth"
    torch.save(policy_net.state_dict(), model_file)
    #print(f"Model saved to {model_file}")

    writer.close()
    env.close()

if __name__ == "__main__":
    train_dqn(  )
