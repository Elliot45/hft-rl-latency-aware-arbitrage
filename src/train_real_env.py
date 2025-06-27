from stable_baselines3 import PPO
from src.real_hft_env import RealHFTEnv
import os

SAVE_PATH = "results/checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)


def main():
    env = RealHFTEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save(os.path.join(SAVE_PATH, "ppo_real_env_model"))
    print("Model trained and saved!")


if __name__ == "__main__":
    main()
