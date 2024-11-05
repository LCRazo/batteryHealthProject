import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()

        # Define the number of battery cells.foundational for creating an accurate battery pack model.
        # Can cells be random?
        self.num_cells = 5

        # Time step (Δt) for discrete updates
        self.time_step = 1

        # Total battery capacity (in milliamp-hours, mAh)
        self.total_capacity = 3400  # Example: 3400 mAh

        # Coulombic efficiency (η), assumed to be 1 for discharge and <= 1 for charge
        self.coulomb_efficiency = 1+.0

        # Initialize internal resistance (R0) and RC branch parameters (Ri, Ci)
        self.R0 = np.array([0.1] * self.num_cells)  # Internal resistance for each cell
        self.Ri = np.array([0.03] * self.num_cells)  # Resistances in the RC branches
        self.Ci = np.array([750] * self.num_cells)  # Capacitances in the RC branches

        # Define action and observation spaces
        # Action space: Each cell has a binary action (0 = switch off, 1 = switch on)
        self.action_space = spaces.MultiBinary(self.num_cells)

        # Observation space: SoC (State of Charge) and switch state for each cell
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cells, 2), dtype=np.float32)

        # Initialize the state of charge (SoC) for each cell and switch states
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Random SoC between 40% and 60%
        self.switch_state = np.zeros(self.num_cells)  # All switches initially off (0)

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state at the start of each episode
            super().reset(seed=seed)
            self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Reset SoC for each cell
            self.switch_state = np.zeros(self.num_cells)  # Reset all switches to off
            self.time_step = 0  # Reset time step counter
            return self._get_obs(), {}  # Return the initial observation and an empty info dictionary

    def step(self, action):
        # Apply the action from the agent and update the environment state

        # Update the switch state based on the agent's action (1 = switch on, 0 = switch off)
        self.switch_state = action

        # Update the state of charge (SoC) for each cell based on whether its switch is on or off
        for i in range(self.num_cells):
            if action[i] == 1:  # If the switch for the cell is on
                current = self._calculate_current(i)  # Calculate the current using the RC model
                # Update SoC using the formula: ΔSoC = -(η * Δt / Q) * i, where i is the current
                self.soc[i] -= (self.coulomb_efficiency * self.time_step / self.total_capacity) * current

                # Optional: Add limits on how much SoC can be discharged to prevent rapid depletion
                self.soc[i] = max(self.soc[i], 0)  # Prevent SoC from going below 0
            else:
                current = 0  # No current flows if the switch is off

        # Calculate the reward based on the SoC variance and switch usage
        # Update reward implementing the temp
        reward = -np.var(self.soc) - np.sum(action) * 0.1  # SoC variance penalty + switch usage penalty

        # Consider adding penalties for extreme SoC values
        if np.any(self.soc <= 0.1):  # Penalize if any cell's SoC is very low
            reward -= 5
        elif np.any(self.soc >= 0.9):  # Penalize if any cell's SoC is very high
            reward -= 5

        # Determine if the episode is done
        done = np.any(self.soc <= 0) or np.any(self.soc >= 1)  # Done if SoC reaches a limit (0 or 100%)

        # Truncated is False as this is not a time-limited environment (modify if needed)
        truncated = False

        # Return the new observation, reward, done, truncated, and info
        return self._get_obs(), reward, done, truncated, {}

    def _calculate_current(self, i):
        # Calculate the current for a given cell using the RC branch model

        # RC branch model: current decays exponentially over time due to the resistances and capacitances
        exp_term = np.exp(-1 / (self.Ri[i] * self.Ci[i]))  # Exponential decay term
        ir_next = (1 - exp_term) * (self.soc[i] - exp_term * self.Ri[i])  # Next current in the RC branch
        return ir_next  # Return the calculated current

    def _get_obs(self):
        # Return the current observation, including SoC and switch states
        return np.column_stack((self.soc, self.switch_state))  # Combine SoC and switch state into a single observation

    def render(self, mode='human'):
        # Render the current state of the environment (for debugging or monitoring)
        print(f'SoC: {self.soc}, Switches: {self.switch_state}')  # Display the SoC and switch states
