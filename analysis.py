import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the logs
log_file = "logs/battery_dqn_logs_20241123-233522.csv"  # Replace with the correct log path
df = pd.read_csv(log_file)

# Convert SOC column back to list
df["soc"] = df["soc"].apply(eval)

# Group SOC values by episode
grouped = df.groupby("episode")["soc"].apply(list)

# Calculate average SOC per episode for each cell
num_cells = 5  # Number of cells
soc_per_cell = {f"Cell {i}": [] for i in range(num_cells)}

for episode, soc_list in grouped.items():
    for cell_idx in range(num_cells):
        avg_soc = sum([soc[cell_idx] for soc in soc_list]) / len(soc_list)
        soc_per_cell[f"Cell {cell_idx}"].append(avg_soc)

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