import gymnasium as gym
import numpy as np
import pandas as pd

class RealHFTEnv(gym.Env):
    def __init__(self, csv_path="data/ethusdt_ticks.csv", max_steps=500):
        super(RealHFTEnv, self).__init__()

        self.data = pd.read_csv(csv_path)
        self.max_steps = min(max_steps, len(self.data))
        self.current_step = 0

        # Paramètres internes
        self.inventory = 0
        self.cash = 0.0

        # Observation: [price, inventory]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32
        )

        # Action: 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0
        self.cash = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row["price"]

        # Action
        if action == 1:  # Buy
            self.inventory += 1
            self.cash -= price
        elif action == 2:  # Sell
            self.inventory -= 1
            self.cash += price

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # PnL = cash + mark-to-market
        mtm_value = self.cash + self.inventory * price
        reward = mtm_value

        obs = self._get_obs()
        info = {"price": price, "inventory": self.inventory, "cash": self.cash}
        return obs, reward, done, False, info

    def _get_obs(self):
        price = self.data.iloc[self.current_step]["price"]
        return np.array([price, self.inventory], dtype=np.float32)

    def render(self):
        print(f"Step: {self.current_step} | Inventory: {self.inventory} | Cash: {self.cash:.2f}")
