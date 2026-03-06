import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from load_data import fetch_and_preprocess_data
from trading_env import TradingEnv
from stable_baselines3.common.vec_env import VecFrameStack
import random
import numpy as np

from typing import Callable

import torch # <--- Add this!

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Lock in the seed
SEED = 42 
set_global_seed(SEED)

# Helper function for dynamic learning rate
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train_models():
    # 1. Fetch Training Data (2015 to the end of 2024)
    print("Fetching training data...")
    train_df = fetch_and_preprocess_data(ticker="SPY", start_date="2015-01-01", end_date="2025-01-01")
    
    # 2. Setup the Environment
    env = DummyVecEnv([lambda: TradingEnv(df=train_df)])
    # NEW: Stack the last 10 days of observations so the agent sees momentum!
    env = VecFrameStack(env, n_stack=10)
    os.makedirs("models", exist_ok=True)

    policy_kwargs = dict(net_arch=[128, 128]) # Doubles the size of the AI's brain

    # Force A2C to explore more
    print("\n--- Training A2C Agent ---")
    model_a2c = A2C("MlpPolicy", env, 
        ent_coef=0.05, 
        policy_kwargs=policy_kwargs, # <--- NEW
        learning_rate=linear_schedule(0.001), 
        seed=SEED,
        verbose=0)
    model_a2c.learn(total_timesteps=500000) 
    model_a2c.save("models/a2c_trading_agent")
    print("A2C Model saved!")

    # Force PPO to explore more
    print("\n--- Training PPO Agent ---")
    # Update your models to use the schedule and optimized batch sizes
    model_ppo = PPO("MlpPolicy", env, 
        ent_coef=0.01, 
        policy_kwargs=policy_kwargs, # <--- NEW
        learning_rate=linear_schedule(0.001), # Starts at 0.001, decays to 0
        batch_size=128, 
        seed=SEED,
        verbose=0)
    model_ppo.learn(total_timesteps=500000)
    model_ppo.save("models/ppo_trading_agent")
    print("PPO Model saved!")

if __name__ == "__main__":
    train_models()