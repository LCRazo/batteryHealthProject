import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the logs
log_file = "logs/battery_dqn_logs_20241201-221728.csv"  # Replace with the correct log path
df = pd.read_csv(log_file)

# Convert SOC, temperature, voltage, and actions columns back to list
df["soc"] = df["soc"].apply(eval)
df["actions"] = df["actions"].apply(eval)
df["temperature"] = df["temperature"].apply(eval)
df["voltage"] = df["voltage"].apply(eval)

# Group SOC and Voltage values by episode
grouped_soc = df.groupby("episode")["soc"].apply(list)
grouped_voltage = df.groupby("episode")["voltage"].apply(list)
grouped_actions = df.groupby("episode")["actions"].apply(list)

# Number of cells
num_cells = 5

# Plot 1: Switching Behavior
time_steps = np.arange(0, 6001, 100)  # Adjust time range based on your logs
switching_behavior = {f"Cell {i+1}": [] for i in range(num_cells)}

# Aggregate switch states over episodes
for episode, actions_list in grouped_actions.items():
    for cell_idx in range(num_cells):
        switches = [action[cell_idx] for action in actions_list]
        switching_behavior[f"Cell {cell_idx+1}"].extend(switches)

# Plot switching behavior
plt.figure(figsize=(10, 6))
for cell, switches in switching_behavior.items():
    plt.plot(time_steps[:len(switches)], switches[:len(time_steps)], label=cell)
plt.title("Switching Behavior of "
          ""
          "Cells")
plt.xlabel("Time (s)")
plt.ylabel("Switches")
plt.yticks([0, 1], ["Off", "On"])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot 2: Voltage Variation
voltage_per_cell = {f"Cell {i+1}": [] for i in range(num_cells)}

# Aggregate voltage data over episodes
for episode, voltage_list in grouped_voltage.items():
    for cell_idx in range(num_cells):
        voltages = [voltage[cell_idx] for voltage in voltage_list]
        voltage_per_cell[f"Cell {cell_idx+1}"].extend(voltages)

# Plot voltage variation
plt.figure(figsize=(10, 6))
for cell, voltages in voltage_per_cell.items():
    plt.plot(time_steps[:len(voltages)], voltages[:len(time_steps)], label=cell)
plt.title("Voltage Variation of Cells")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
# Set y-axis limits
plt.ylim(3.0, 4.4)
plt.grid()
plt.tight_layout()
plt.show()

# Plot 3: Smoothed SOC per Episode
# Calculate average SOC per episode for each cell
soc_per_cell = {f"Cell {i+1}": [] for i in range(num_cells)}
for episode, soc_list in grouped_soc.items():
    for cell_idx in range(num_cells):
        avg_soc = sum([soc[cell_idx] for soc in soc_list]) / len(soc_list)
        soc_per_cell[f"Cell {cell_idx+1}"].append(avg_soc)

# Smooth the SOC data using a rolling average
window_size = 50  # Adjust window size for smoothing
smoothed_soc_per_cell = {
    cell: pd.Series(values).rolling(window=window_size, min_periods=1).mean()
    for cell, values in soc_per_cell.items()
}

# Combine SOC data into a single plot
plt.figure(figsize=(10, 6))
for cell, soc_values in smoothed_soc_per_cell.items():
    plt.plot(soc_values, label=cell)
plt.title("State of Charge (SOC) per Cell by Episode (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("SOC (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
#test
# Plot 4: Horizontal Bar Plots for Selected Episodes
# Identify episodes where the batteries are charging or discharging
# Function to generate SOC data for an episode

def generate_soc_data(num_cells, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, num_cells)

# Generate SOC data for 3 episodes (Initial, Discharging, Charging)
grouped_soc = [
    generate_soc_data(5, 0.1, 0.4),  # Initial state (SOC between 10% and 50%)
    generate_soc_data(5, 0.15, 0.25),  # Discharging episode (SOC between 10% and 30%)
    generate_soc_data(5, 0.5, 0.6),   # Charging episode (SOC between 40% and 60%)
]

# Identify charging and discharging episodes
charging_episodes = []
discharging_episodes = []

for episode, soc_values in enumerate(grouped_soc):
    soc_values = np.array(soc_values)
    if np.any(soc_values > 0.5):  # Charging condition (SOC > 40%)
        charging_episodes.append((episode, soc_values))
    if np.any((soc_values >= 0.1) & (soc_values <= 0.3)):  # Discharging condition (SOC between 15-30%)
        discharging_episodes.append((episode, soc_values))

# Function to find the most balanced episode based on SOC variance
def find_most_balanced_episode(episodes):
    if not episodes:
        return None, None
    variances = [np.var(soc) for _, soc in episodes]
    min_variance_index = np.argmin(variances)  # Episode with minimum SOC variance
    return episodes[min_variance_index]

# Initial state
initial_state = np.array(grouped_soc[0])

# Most balanced discharging and charging episodes
discharging_episode, discharging_values = find_most_balanced_episode(discharging_episodes)
charging_episode, charging_values = find_most_balanced_episode(charging_episodes)

# Prepare SOC snapshots (limit to top 5 cells by SOC value)
soc_snapshots = {
    "Initial State": np.sort(initial_state)[-5:],  # Top 5 SOC values
    "Discharging": np.sort(discharging_values)[-5:] if discharging_values is not None else np.zeros(5),
    "Charging": np.sort(charging_values)[-5:] if charging_values is not None else np.zeros(5),
}

# Plotting
fig, axes = plt.subplots(1, len(soc_snapshots), figsize=(15, 5), sharey=True)
colors = ["blue", "red", "green"]

for idx, (label, soc_values) in enumerate(soc_snapshots.items()):
    ax = axes[idx]
    ax.barh(range(1, 6), soc_values, color=colors[idx], alpha=0.7)  # Top 5 cells
    ax.set_title(label)
    ax.set_xlim(0, 1)  # SOC range from 0 to 1 (0-100%)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"Cell {i}" for i in range(1, 6)])
    ax.set_xlabel("SOC")

# Shared ylabel
axes[0].set_ylabel("Cells")
plt.tight_layout()
plt.show()

# Output selected episodes for debugging
print("SOC Snapshots for plotting:")
for label, values in soc_snapshots.items():
    print(f"{label}: {values}")

if discharging_values is not None:
    print(f"Discharging Episode: {discharging_episode}, SOC Values: {discharging_values}")
else:
    print("No discharging episode found.")

if charging_values is not None:
    print(f"Charging Episode: {charging_episode}, SOC Values: {charging_values}")
else:
    print("No charging episode found.")


# Plot 5: Temperature Variation
temperature_per_cell = {f"Cell {i+1}": [] for i in range(num_cells)}

# Aggregate temperature data over episodes
grouped_temperature = df.groupby("episode")["temperature"].apply(list)
for episode, temp_list in grouped_temperature.items():
    for cell_idx in range(num_cells):
        temperatures = [temp[cell_idx] for temp in temp_list]
        temperature_per_cell[f"Cell {cell_idx+1}"].extend(temperatures)

# Plot temperature variation
plt.figure(figsize=(10, 6))
for cell, temperatures in temperature_per_cell.items():
    plt.plot(time_steps[:len(temperatures)], temperatures[:len(time_steps)], label=cell)
plt.title("Temperature Variation of Cells")
plt.xlabel("Time-steps")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Plot #6 best switching episode
# Define target balance SOC (e.g., 50%)
target_soc = 0.6

# Variables to track the most balanced episode
most_balanced_episode = None
lowest_soc_deviation = float("inf")

# Analyze SOC balance for each episode
soc_balance = {}
for episode, actions_list in grouped_actions.items():
    # Assuming SOC values are part of the episode data
    # Extract SOC values (modify this part based on your SOC representation)
    soc_values = [np.mean(action) for action in actions_list]  # Average SOC for each timestep
    avg_soc = np.mean(soc_values)  # Mean SOC for the entire episode
    deviation = abs(avg_soc - target_soc)  # Deviation from the target SOC

    soc_balance[episode] = {"avg_soc": avg_soc, "deviation": deviation}

    # Update the most balanced episode based on SOC deviation
    if deviation < lowest_soc_deviation:
        most_balanced_episode = episode
        lowest_soc_deviation = deviation

# Extract actions for the most balanced episode
if most_balanced_episode is not None:
    most_balanced_actions = np.array(grouped_actions[most_balanced_episode])
    time_steps = np.arange(most_balanced_actions.shape[0])  # Generate time steps

    # Plot switching behavior
    plt.figure(figsize=(12, 8))
    for cell_idx in range(most_balanced_actions.shape[1]):
        plt.step(time_steps, most_balanced_actions[:, cell_idx], label=f"Cell {cell_idx + 1}", where='mid')

    plt.xlabel("Time (s)")
    plt.ylabel("Switches")
    plt.title("Switching Behavior of Cells (Most Balanced SOC Episode)")
    plt.yticks([0, 1], labels=["Off", "On"])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Most Balanced Episode: {most_balanced_episode}")
    print(f"Average SOC: {soc_balance[most_balanced_episode]['avg_soc']}")
else:
    print("No suitable episodes found for SOC balance.")

#epidose with less switching
# Filtering actions to 0-6000 seconds
time_limit = 6000
filtered_grouped_actions = {ep: actions for ep, actions in grouped_actions.items() if len(actions) <= time_limit}

# Find best switching episode within the range
best_switching_episode = None
lowest_switch_variance = float("inf")

for episode, actions_list in filtered_grouped_actions.items():
    switch_counts = []
    for cell_idx in range(len(actions_list[0])):
        switches = [action[cell_idx] for action in actions_list]
        switch_count = np.sum(np.abs(np.diff(switches)))
        switch_counts.append(switch_count)

    variance = np.var(switch_counts)
    if variance < lowest_switch_variance:
        best_switching_episode = episode
        lowest_switch_variance = variance

# Ensure a valid best episode
if best_switching_episode is not None:
    best_episode_actions = np.array(filtered_grouped_actions[best_switching_episode])
    time_steps = np.linspace(0, 6000, best_episode_actions.shape[0])

    # Plot switching behavior
    plt.figure(figsize=(12, 8))
    for cell_idx in range(best_episode_actions.shape[1]):
        plt.step(time_steps, best_episode_actions[:, cell_idx], label=f"Cell {cell_idx + 1}", where='mid')

    plt.xlabel("Time (s)")
    plt.ylabel("Switches")
    plt.title("Switching Behavior of Cells (Best Episode: 0-6000 seconds)")
    plt.yticks([0, 1], labels=["Off", "On"])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Best Switching Episode: {best_switching_episode}")
else:
    print("No suitable episodes found in the range 0-6000 seconds.")


