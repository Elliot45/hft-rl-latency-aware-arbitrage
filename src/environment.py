import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class HFTEnv(gym.Env):
    """
    Environnement HFT simplifié avec latence d'exécution.
    """
    def __init__(self):
        super(HFTEnv, self).__init__()

        # Observation : [bid, ask, spread, imbalance, latency]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Actions : 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # Paramètres internes
        self.max_steps = 1000
        self.step_count = 0
        self.latency_window = [0, 1, 2]  # latence en steps
        self.pending_actions = []  # file d’attente des actions à exécuter avec latence

        self.price = 100.0
        self.inventory = 0
        self.cash = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.price = 100.0
        self.inventory = 0
        self.cash = 0
        self.step_count = 0
        self.pending_actions.clear()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        bid = self.price - 0.5
        ask = self.price + 0.5
        spread = ask - bid
        imbalance = np.random.uniform(-1, 1)
        latency = random.choice(self.latency_window)

        # Ajouter l’action avec délai d’exécution
        self.pending_actions.append((latency, action))

        # Exécuter actions en attente dont la latence est expirée
        executed_actions = []
        for i, (delay, a) in enumerate(self.pending_actions):
            if delay == 0:
                executed_actions.append(a)
        self.pending_actions = [(d-1, a) for d, a in self.pending_actions if d > 0]

        for exec_action in executed_actions:
            if exec_action == 0:  # Sell
                self.cash += self.price
                self.inventory -= 1
            elif exec_action == 2:  # Buy
                self.cash -= self.price
                self.inventory += 1

        # Mouvements aléatoires du prix
        self.price += np.random.randn() * 0.1

        unrealized_pnl = self.inventory * self.price
        reward = self.cash + unrealized_pnl

        done = self.step_count >= self.max_steps
        obs = np.array([bid, ask, spread, imbalance, latency], dtype=np.float32)

        return obs, reward, done, False, {}

    def _get_obs(self):
        bid = self.price - 0.5
        ask = self.price + 0.5
        spread = ask - bid
        imbalance = np.random.uniform(-1, 1)
        latency = random.choice(self.latency_window)
        return np.array([bid, ask, spread, imbalance, latency], dtype=np.float32)
