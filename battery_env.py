import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG to capture all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.FileHandler("battery_env.log"),  # Save logs to a file named 'battery_env.log'
        logging.StreamHandler()  # Also print logs to the console
    ]
)

# Create a logger instance
logger = logging.getLogger("BatteryEnvLogger")

# Example usage in your environment
logger.debug("This is a debug message for testing log saving.")
logger.info("Information message.")
logger.warning("Warning message.")
logger.error("Error message.")


class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        # logger.debug("BatteryEnv initialized")

        # Number of battery cells
        self.num_cells = 5

        # Time step
        self.time_step = 1  # 1 second per step
        self.current_time = 0  # Track current simulation time (in seconds)

        # Load profile durations (in seconds)
        self.charge_duration = 4373  # Slightly longer due to inefficiencies
        self.discharge_duration = 4165
        self.rest_duration = 100
        self.max_time = self.charge_duration + self.rest_duration + self.discharge_duration

        # Total battery capacity (in mAh)
        self.total_capacity = 3400

        # Voltage for each battery cell
        self.voltage = np.full(self.num_cells, 3.6)
        self.desired_voltage = 3.6  # Set to the desired voltage value

        self.coulomb_efficiency = 1.0
        self.alpha = 0.1

        # Thermal parameters
        self.r_th_in = 0.01  # Internal thermal resistance
        self.r_th_out = 0.02  # External thermal resistance
        self.t_amb = 298.15  # Ambient temperature in Kelvin (25°C)
        self.c_th = 0.1  # Thermal capacitance
        self.t_cell = np.random.uniform(290, 300, self.num_cells)  # Initial temperatures

        # Action and observation space
        self.action_space = spaces.MultiBinary(self.num_cells)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cells, 4), dtype=np.float32)

        # Initial SOC for cells
        self.soc = np.clip(np.random.uniform(0.4, 0.6, self.num_cells), 0.1, 0.9)  # Random SOC between 40% and 60%
        self.switch_state = np.zeros(self.num_cells)  # Switches initially off

        # Initialize resistance and capacitance parameters dynamically
        self.parameters = self.compute_dynamic_parameters_vectorized(self.soc)

        # Initialize metrics for logging
        self.metrics = {
            "soc_per_cell": {i: [] for i in range(self.num_cells)},
            "temperature_per_cell": {i: [] for i in range(self.num_cells)},
            "switch_state_per_cell": {i: [] for i in range(self.num_cells)},
            "reward_per_cell": {i: [] for i in range(self.num_cells)},
            "max_temperature_per_cell": {i: -np.inf for i in range(self.num_cells)},  # Initialized to very low
            "voltage_per_cell": {i: [] for i in range(self.num_cells)}
        }

        self.current_episode = 0  # Tracks the current episode
        self.total_episodes = 10_000  # Total planned episodes

    def get_load_current(self):
        """Determine the load current based on the current time and the load profile."""
        if self.current_time < self.charge_duration:  # Charge phase
            load_current = -2.35
            phase = "Charge"
        elif self.charge_duration <= self.current_time < (self.charge_duration + self.rest_duration):  # Rest phase
            load_current = 0.0
            phase = "Rest"
        elif (self.charge_duration + self.rest_duration) <= self.current_time < self.max_time:  # Discharge phase
            load_current = 2.35
            phase = "Discharge"
        else:  # End of the profile
            load_current = 0.0
            phase = "End"

        # Calculate the number of active cells for logging (if needed)
        # num_active_cells = np.sum(self.switch_state == 1)  # Count active cells

        # Add logging for verification
        # logger.debug(
        # f"Current time: {self.current_time}, Phase: {phase}, Active cells: {num_active_cells}, Load Current: {load_current}")

        # Return the load current and phase as the last step
        return load_current, phase

    def compute_dynamic_parameters_vectorized(self, soc_array):
        """Compute internal battery parameters based on SOC."""
        r0_min, r0_max = 0.1063, 0.1325  # Internal resistance (Ω)
        r1_min, r1_max = 0.0257, 0.0498  # RC branch resistance 1 (Ω)
        c1_max = 1161.67  # Maximum capacitance 1 (F)
        r2_min, r2_max = 0.0078, 0.0096  # RC branch resistance 2 (Ω)
        c2_max = 791.77  # Maximum capacitance 2 (F)

        r0 = r0_min + (r0_max - r0_min) * (1 - soc_array)
        r1 = r1_min + (r1_max - r1_min) * (1 - soc_array)
        c1 = c1_max * soc_array
        r2 = r2_min + (r2_max - r2_min) * (1 - soc_array)
        c2 = c2_max * soc_array

        return [{"r0": r0[i], "r1": r1[i], "c1": c1[i], "r2": r2[i], "c2": c2[i]} for i in range(len(soc_array))]

    def step(self, action):
        self.switch_state = np.array(action)  # Ensure action is a NumPy array
        active_cells = self.switch_state == 1  # Identify active cells

        # Increment the simulation time
        self.current_time += self.time_step
        if self.current_time >= self.max_time:
            # logger.debug(f"Resetting current_time to 0 after reaching max_time: {self.max_time}")
            self.current_time = 0

        # Get the load current and phase
        load_current, phase = self.get_load_current()

        # Calculate the total load and balance it across cells
        total_load = load_current * np.sum(active_cells)  # Total load based on active cells
        if np.sum(active_cells) > 0:
            load_per_cell = total_load / self.num_cells  # Distribute load evenly across all cells
            self.soc -= load_per_cell / self.total_capacity  # Apply balanced load to SOC
        # else:
        # logger.debug("No active cells, skipping load balancing.")

        # Calculate the number of active cells
        num_active_cells = np.sum(active_cells)
        if num_active_cells > 0:
            load_current_per_cell = load_current / num_active_cells
        else:
            load_current_per_cell = 0.0

        currents = np.zeros(self.num_cells)  # Initialize currents array
        currents[active_cells] = load_current_per_cell

        # Update SOC and temperatures
        inactive_cells = ~active_cells
        self.soc[inactive_cells] -= 0.00001  # Idle discharge
        self.soc -= 0.00002  # Baseline discharge
        self.soc = np.clip(self.soc, 0.1, 0.9)  # Clip SOC

        # Update the temperatures of the cells
        self._update_temperature(currents)

        # Calculate reward and check termination
        _, _, _, _, _, _, reward, done = self._calculate_reward()  # Unpack only what is needed

        # Distribute reward contributions per cell
        for i in range(self.num_cells):
            self.metrics["reward_per_cell"][i].append(reward / self.num_cells)

        v_batt_values = np.zeros(self.num_cells)

        # Check for active cells
        if np.any(active_cells):  # Only proceed if there are active cells
            active_indices = np.where(active_cells)[0]  # Convert boolean mask to integer indices
            currents[active_cells], v_batt_values[active_cells] = self._calculate_currents(active_cells)
            self.voltage[active_cells] = v_batt_values[active_cells]  # Update voltages
            self._update_soc(currents[active_cells], active_indices)  # Pass active currents and indices

            updated_parameters = self.compute_dynamic_parameters_vectorized(self.soc[active_indices])
            for idx, param in zip(active_indices, updated_parameters):
                self.parameters[idx] = param

        # Apply load current to all cells
        if np.any(active_cells):
            currents = load_current * np.ones(self.num_cells)  # Apply the load profile
            self._update_soc(currents, np.arange(self.num_cells))  # Update SOC for all cells

        # Apply idle and baseline discharge rates
        idle_discharge = 0.00001  # Reduced idle discharge rate
        baseline_discharge = 0.00002  # Reduced baseline discharge rate
        inactive_cells = ~active_cells
        self.soc[inactive_cells] -= idle_discharge
        self.soc -= baseline_discharge

        # Clip SOC to ensure it's within the safe range
        self.soc = np.clip(self.soc, 0.1, 0.9)

        # Distribute reward contributions per cell
        for i in range(self.num_cells):
            self.metrics["reward_per_cell"][i].append(reward / self.num_cells)  # Divide reward equally
        # Determine if the episode is done
        if np.any(self.soc <= 0.1):
            # logger.debug(f"Environment done: SOC below threshold. SOC: {self.soc}")
            done = True
        elif np.any(self.t_cell > 330):
            # logger.debug(f"Environment done: Temperature exceeded limit. Temperature: {self.t_cell}")
            done = True
        elif self.current_time >= self.max_time:
            # logger.debug(f"Environment done: Current time reached max time. Current time: {self.current_time}")
            done = True
        else:
            done = False

        # Return the observation, reward, done, and info dictionary
        return self._get_obs(), reward, done, False, {}

    def _calculate_currents(self, active_cells):
        # Extract parameters for active cells
        r0_values = np.array([p["r0"] for p in self.parameters])[active_cells]
        r1_values = np.array([p["r1"] for p in self.parameters])[active_cells]
        c1_values = np.array([p["c1"] for p in self.parameters])[active_cells]
        soc_values = self.soc[active_cells]
        switch_state_values = self.switch_state[active_cells]
        voltage_values = self.voltage[active_cells]

        # Calculate open-circuit voltage for active cells
        v_ocv_values = self._calculate_ocv(soc_values)

        # Voltage under load
        v_batt_values = v_ocv_values + r0_values * switch_state_values

        # Primary battery current
        i_batt_values = (voltage_values - v_batt_values) / r0_values

        # RC branch current contribution
        exp_terms = np.exp(-1 / (r1_values * c1_values))  # Exponential decay terms
        rc_currents = (1 - exp_terms) * soc_values

        # Total currents
        total_currents = i_batt_values + rc_currents

        # Dynamic clamping based on training progress
        training_progress = self.current_episode / self.total_episodes if hasattr(self, "current_episode") else 0
        max_clamp = 5 + 10 * training_progress  # Scale between 5 and 15
        current_clamp = np.minimum(max_clamp, 15)  # Cap at 15
        total_currents = np.clip(total_currents, -current_clamp, current_clamp)

        return total_currents, v_batt_values

    def _update_temperature(self, currents):
        r0_values = np.array([p["r0"] for p in self.parameters])  # Extract r0 for all cells
        q_gen = np.maximum(0, np.power(currents, 2) * r0_values)  # Joule heating for all cells
        delta_t = np.clip(
            (q_gen / self.c_th) + (self.t_amb - self.t_cell) / self.r_th_out,
            -5, 5
        )  # Temperature change for all cells
        self.t_cell = np.clip(self.t_cell + delta_t, self.t_amb, 350.0)  # Update and clamp temperatures
        return delta_t

    @staticmethod
    def _calculate_ocv(soc):
        v_min, v_max = 3.3, 4.1  # Voltage range corresponding to SOC range 10%-90%
        return v_min + (v_max - v_min) * soc  # Linear relationship for simplicity

    def _update_soc(self, active_currents, active_indices):
        # Open-circuit voltage (vectorized for active cells)
        v_ocv = self._calculate_ocv(self.soc[active_indices])

        # Effective voltage (average between current and open-circuit voltage)
        effective_voltage = (self.voltage[active_indices] + v_ocv) / 2

        # Use active_currents to calculate SOC change
        delta_soc = (
                            effective_voltage * self.coulomb_efficiency * self.time_step / self.total_capacity
                    ) * np.abs(active_currents)
        self.soc[active_indices] -= delta_soc  # Decrease SOC for discharging

        delta_soc = (
                            effective_voltage * self.coulomb_efficiency * self.time_step / self.total_capacity
                    ) * np.abs(active_currents)
        self.soc[active_indices] -= delta_soc

        # Clip SOC to ensure it's within the safe range
        self.soc = np.clip(self.soc, 0.1, 0.9)

        # Update the voltage for all cells
        for idx in active_indices:
            self.voltage[idx] = self._calculate_ocv(self.soc[idx])

    def _calculate_reward(self):
        # SOC Variance Reward
        soc_variance = np.var(self.soc)
        normalized_soc_variance = soc_variance / (np.mean(self.soc) + 1e-8)  # Normalize SOC variance
        r_soc = -np.log(normalized_soc_variance + 1e-8)

        # Temperature Reward (Variance + Optimal Range)
        temperature_variance = np.var(self.t_cell)
        mean_temp = np.mean(self.t_cell)
        normalized_temp_variance = max(temperature_variance / (mean_temp + 1e-8), 1e-8)
        r_temp = -normalized_temp_variance ** 2
        r_temp /= max(abs(r_temp), 1e-8)  # Normalize r_temp for variance

        # Add Positive Reward for Optimal Temperature Range
        optimal_temp_min = 288  # 15°C in Kelvin
        optimal_temp_max = 318  # 45°C in Kelvin
        if np.all((self.t_cell >= optimal_temp_min) & (self.t_cell <= optimal_temp_max)):
            r_temp += 5  # Add a positive reward if all cells are in the optimal range

        # Usable Capacity Reward
        avg_soc = np.mean(self.soc)
        temp_deviation = np.mean(np.abs(self.t_cell - 298.15))
        usable_capacity_reward = avg_soc * np.exp(-temp_deviation / 10)
        usable_capacity_reward /= max(abs(usable_capacity_reward), 1e-8)  # Adjusting rewards

        # Switching Penalty
        switch_count = np.sum(np.abs(np.diff(self.switch_state)))
        threshold = 3
        excessive_switching = max(0, switch_count - threshold)
        r_switch = -self.alpha * excessive_switching
        r_switch /= max(abs(r_switch), 1e-8)  # Adjusting rewards

        # Penalty for extreme conditions
        penalty = 0
        if np.any(self.t_cell > 330):  # Check if temperature exceeds the threshold
            penalty -= np.exp((self.t_cell.max() - 330) / 10)

        if np.any(self.soc < 0.1):  # Check if SOC goes below the threshold
            penalty -= np.exp((0.1 - self.soc.min()) / 0.01)

        # Normalize penalty only if it's not zero
        if penalty != 0:
            penalty /= np.max(np.abs(penalty) + 1e-8)

        # SOC Balance Reward
        r_balance = -np.log(normalized_soc_variance + 1e-8)

        # Final Reward
        final_reward = (
                0.5 * r_soc +
                0.2 * r_temp +  # Updated r_temp includes both variance and optimal range reward
                0.15 * r_balance +
                0.1 * r_switch +
                penalty +
                0.4 * usable_capacity_reward
        )

        # Done flag
        done = np.any(self.soc <= 0.1) or np.any(self.t_cell > 330)

        # Return all the calculated values
        return r_soc, r_temp, usable_capacity_reward, r_switch, penalty, r_balance, final_reward, done

        # Logging the reward components
        #logger.debug(f"Reward Breakdown:")
        #logger.debug(f"  - r_soc (SOC Variance Reward): {r_soc:.4f}")
        #logger.debug(f"  - r_temp (Temperature Variance Reward): {r_temp:.4f}")
        #logger.debug(f"  - Usable Capacity Reward: {usable_capacity_reward:.4f}")
        #logger.debug(f"  - r_switch (Switching Penalty): {r_switch:.4f}")
        #logger.debug(f"  - Penalty (Extreme Conditions): {penalty:.4f}")
        #logger.debug(f"  - r_balance (SOC Balance Reward): {r_balance:.4f}")
        #logger.debug(f"  - Final Reward: {reward:.4f}")


    def reset(self, seed=None, options=None):
        self.current_episode += 1  # Increment episode count
        self.current_time = 0  # Reset the time for the load profile

        super().reset(seed=seed)
        # logger.debug(f"Reset called. Current episode: {self.current_episode}, Current time reset to 0.")
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Reset SOC
        # logger.debug(f"Reset called. Initialized SOC: {self.soc}")
        self.switch_state = np.zeros(self.num_cells)  # Reset switches
        self.t_cell = np.random.uniform(290, 300, self.num_cells)  # Reset temperatures
        self.parameters = self.compute_dynamic_parameters_vectorized(self.soc)  # Reset parameters

        # Reset metrics
        self.metrics = {
            "soc_per_cell": {i: [] for i in range(self.num_cells)},
            "temperature_per_cell": {i: [] for i in range(self.num_cells)},
            "switch_state_per_cell": {i: [] for i in range(self.num_cells)},
            "reward_per_cell": {i: [] for i in range(self.num_cells)},
            "max_temperature_per_cell": {i: -np.inf for i in range(self.num_cells)},
            "voltage_per_cell": {i: [] for i in range(self.num_cells)},
        }

        return self._get_obs(), {}

    def _get_obs(self):
        return np.column_stack((self.soc, self.switch_state, self.t_cell, self.voltage))

    def render(self, mode='human'):
        print(
            f"Reward: {reward}, SOC: {self.soc}, Temp: {self.t_cell}, Switch State: {self.switch_state}, Voltage: {self.voltage}, Usable Capacity Reward: {usable_capacity_reward}")