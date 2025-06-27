from src.environment import HFTEnv
import numpy as np

def test_environment_step():
    env = HFTEnv()
    obs = env.reset()
    
    assert isinstance(obs, np.ndarray), "Observation should be a NumPy array"
    assert obs.shape == (5,), f"Expected observation shape (5,), got {obs.shape}"

    for _ in range(10):
        action = env.action_space.sample()
        new_obs, reward, done, _, info = env.step(action)

        assert isinstance(new_obs, np.ndarray), "New observation should be a NumPy array"
        assert new_obs.shape == (5,), f"New observation shape mismatch: got {new_obs.shape}"
        assert isinstance(reward, float), "Reward should be a float"
        assert isinstance(done, bool), "Done should be a boolean"

    print("Environment test passed.")

if __name__ == "__main__":
    test_environment_step()
