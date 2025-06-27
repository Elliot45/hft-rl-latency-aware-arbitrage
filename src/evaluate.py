from stable_baselines3 import PPO
from src.environment import HFTEnv
from src.baselines import random_strategy, buy_and_hold
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

def evaluate_model(model_path="results/checkpoints/ppo_hft_model", n_steps=500, plot=True):
    env = HFTEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    rewards = []
    prices = []
    actions = []
    inventory = []
    cash = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        rewards.append(reward)
        prices.append(env.price)
        actions.append(action)
        inventory.append(env.inventory)
        cash.append(env.cash)

        if done:
            break

    print(f"\nPPO Agent — Reward final : {rewards[-1]:.2f} | Inv: {inventory[-1]} | Cash: {cash[-1]:.2f} | Actions : {np.bincount(actions)}")

    if plot:
        plt.plot(rewards, label="PPO Agent")
        return rewards
    return rewards, inventory[-1], cash[-1]

def compare_to_baselines():
    print("\nComparaison des stratégies :")

    # PPO
    rewards_ppo = evaluate_model(plot=False)[0]

    # Random Strategy
    rewards_random, inv_r, cash_r = random_strategy()
    print(f"Random Strategy — Reward final : {rewards_random[-1]:.2f} | Inv: {inv_r} | Cash: {cash_r:.2f}")

    # Buy & Hold
    rewards_bh, inv_bh, cash_bh = buy_and_hold()
    print(f"Buy & Hold — Reward final : {rewards_bh[-1]:.2f} | Inv: {inv_bh} | Cash: {cash_bh:.2f}")

    # Courbe comparée
    plt.plot(rewards_ppo, label="PPO Agent")
    plt.plot(rewards_random, label="Random Strategy")
    plt.plot(rewards_bh, label="Buy & Hold")
    plt.title("Comparaison des rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.savefig("results/plots/reward_{strategy_name}.png")
    plt.show()
    


if __name__ == "__main__":
    compare_to_baselines()
