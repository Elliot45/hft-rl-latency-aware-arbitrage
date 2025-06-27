from src.environment import HFTEnv
import numpy as np

def random_strategy(n_steps=500):
    env = HFTEnv()
    obs, _ = env.reset()
    rewards = []
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    return rewards, env.inventory, env.cash

def buy_and_hold(n_steps=500):
    env = HFTEnv()
    obs, _ = env.reset()
    rewards = []
    bought = False
    for _ in range(n_steps):
        if not bought:
            action = 2  # Buy
            bought = True
        else:
            action = 1  # Hold
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    return rewards, env.inventory, env.cash
