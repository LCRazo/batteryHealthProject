import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()

        # Number of battery cells
        self.num_cells = 5

        # Time step
        self.time_step = 1

        # Total battery capacity (in mAh)
        self.total_capacity = 3400

        # Coulombic efficiency
        self.coulomb_efficiency = 1.0

        # Thermal parameters
        self.r_th_in = 0.01  # Internal thermal resistance
        self.r_th_out = 0.02  # External thermal resistance
        self.t_amb = 298.15  # Ambient temperature in Kelvin (25°C)
        self.c_th = 0.1  # Thermal capacitance
        self.t_cell = np.random.uniform(290, 300, self.num_cells)  # Initial temperatures

        # Action and observation space
        self.action_space = spaces.MultiBinary(self.num_cells)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cells, 3), dtype=np.float32)

        # Initial SOC for cells
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Random SOC between 40% and 60%
        self.switch_state = np.zeros(self.num_cells)  # Switches initially off

        # Initialize resistance and capacitance parameters dynamically
        self.parameters = [self.compute_dynamic_parameters(s) for s in self.soc]

        # Initialize metrics for logging
        self.metrics = {
            "soc_per_cell": {i: [] for i in range(self.num_cells)},
            "temperature_per_cell": {i: [] for i in range(self.num_cells)},
            "switch_state_per_cell": {i: [] for i in range(self.num_cells)},
            "reward_per_cell": {i: [] for i in range(self.num_cells)},
            "max_temperature_per_cell": {i: -np.inf for i in range(self.num_cells)}  # Initialized to very low
        }

    @staticmethod
    def compute_dynamic_parameters(soc):
        """
        Compute dynamic resistance and capacitance values based on SOC.
        """
        r0_min, r0_max = 0.1063, 0.1325  # Internal resistance (Ω)
        r1_min, r1_max = 0.0257, 0.0498  # RC branch resistance 1 (Ω)
        c1_max = 1161.67  # Maximum capacitance 1 (F)
        r2_min, r2_max = 0.0078, 0.0096  # RC branch resistance 2 (Ω)
        c2_max = 791.77  # Maximum capacitance 2 (F)

        r0 = r0_min + (r0_max - r0_min) * (1 - soc)
        r1 = r1_min + (r1_max - r1_min) * (1 - soc)
        c1 = c1_max * soc
        r2 = r2_min + (r2_max - r2_min) * (1 - soc)
        c2 = c2_max * soc

        return {"r0": r0, "r1": r1, "c1": c1, "r2": r2, "c2": c2}

    def step(self, action):
        """
        Updates SOC and temperature of each cell based on the action and calculates the reward.
        """
        self.switch_state = action

        for i in range(self.num_cells):
            if action[i] == 1:  # If the switch for cell `i` is on
                current = self._calculate_current(i)
                self.soc[i] -= (self.coulomb_efficiency * self.time_step / self.total_capacity) * current
                self.soc[i] = max(self.soc[i], 0)  # Prevent negative SOC
                self.t_cell[i] += self._update_temperature(current, i)  # Pass index `i`

                # Dynamically update parameters
                self.parameters[i] = self.compute_dynamic_parameters(self.soc[i])

            # Update max temperature for cell `i`
            self.metrics["max_temperature_per_cell"][i] = max(self.metrics["max_temperature_per_cell"][i], self.t_cell[i])

            # Log metrics for each cell
            self.metrics["soc_per_cell"][i].append(self.soc[i])
            self.metrics["temperature_per_cell"][i].append(self.t_cell[i])
            self.metrics["switch_state_per_cell"][i].append(action[i])

        # Calculate reward and termination
        reward, done = self._calculate_reward()

        # Distribute reward contributions per cell
        for i in range(self.num_cells):
            self.metrics["reward_per_cell"][i].append(reward / self.num_cells)  # Divide reward equally

        return self._get_obs(), reward, done, False, {}

    def _calculate_current(self, i):
        """
        Calculate the current for a given cell using RC branch parameters.
        """
        r0 = self.parameters[i]["r0"]
        r1 = self.parameters[i]["r1"]
        c1 = self.parameters[i]["c1"]

        exp_term = np.exp(-1 / (r1 * c1))
        return (1 - exp_term) * self.soc[i]

    def _update_temperature(self, current, i):
        """
        Update the temperature of the cell based on heat generation.
        """
        q_gen = current ** 2 * self.parameters[i]["r0"]  # Heat generated
        delta_t = q_gen * (self.r_th_in + self.r_th_out) + self.t_amb  # Temperature change
        return delta_t / (1 + self.c_th)

    def _calculate_reward(self):
        """
        Calculate the reward based on SOC variance, temperature variance, and switching penalty.
        """
        # SOC Variance Reward
        soc_variance = np.var(self.soc)
        r_soc = -np.log(soc_variance + 1e-8)  # Logarithmic penalty for SOC variance

        # Temperature Variance Reward
        temperature_variance = np.var(self.t_cell)
        r_temp = -np.log(temperature_variance + 1e-8)

        # Switching Penalty
        switch_penalty = np.sum(self.switch_state)
        r_switch = -0.1 * switch_penalty

        # Penalty for extreme conditions
        penalty = 0
        if np.any(self.soc <= 0.1):  # Low SOC penalty
            penalty -= 10
        if np.any(self.t_cell > 330):  # High temperature penalty
            penalty -= 10

        # Final Reward
        reward = 0.8 * r_soc + 0.2 * r_temp + r_switch + penalty

        # Determine if the episode is done
        done = np.any(self.soc <= 0.1) or np.any(self.t_cell > 330)
        return reward, done

    def _get_obs(self):
        """
        Return the current observation, including SOC, switch state, and temperature.
        """
        return np.column_stack((self.soc, self.switch_state, self.t_cell))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Reset SOC
        self.switch_state = np.zeros(self.num_cells)  # Reset switches
        self.t_cell = np.random.uniform(290, 300, self.num_cells)  # Reset temperatures
        self.parameters = [self.compute_dynamic_parameters(s) for s in self.soc]  # Reset parameters

        # Reset metrics
        self.metrics = {
            "soc_per_cell": {i: [] for i in range(self.num_cells)},
            "temperature_per_cell": {i: [] for i in range(self.num_cells)},
            "switch_state_per_cell": {i: [] for i in range(self.num_cells)},
            "reward_per_cell": {i: [] for i in range(self.num_cells)},
            "max_temperature_per_cell": {i: -np.inf for i in range(self.num_cells)}
        }

        return self._get_obs(), {}

    def render(self, mode='human'):
        """
        Print the current state for debugging.
        """
        print(f"SOC: {self.soc}, Temp: {self.t_cell}, Switch State: {self.switch_state}")