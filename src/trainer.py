from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.environment import HFTEnv
import os

def main():
    env = HFTEnv()
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    os.makedirs("results/checkpoints", exist_ok=True)
    model.save("results/checkpoints/ppo_hft_model")
    print("Entraînement terminé et modèle sauvegardé.")

if __name__ == "__main__":
    main()
