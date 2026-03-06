import os
import argparse
import random
import numpy as np
import torch
from typing import Callable

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from load_data import fetch_and_preprocess_data
from trading_env import TradingEnv

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

def train_models(models_to_train):
    # 1. Fetch Training Data
    print("Fetching training data...")
    train_df = fetch_and_preprocess_data(ticker="SPY", start_date="2015-01-01", end_date="2025-01-01")
    
    # 2. Setup the Environment
    env = DummyVecEnv([lambda: TradingEnv(df=train_df)])
    env = VecFrameStack(env, n_stack=10)
    os.makedirs("models", exist_ok=True)

    policy_kwargs = dict(net_arch=[128, 128])

    # ---------------------------------------------------
    # A2C Training
    # ---------------------------------------------------
    if "a2c" in models_to_train:
        print("\n--- Training A2C Agent ---")
        model_a2c = A2C("MlpPolicy", env, 
            ent_coef=0.05, 
            policy_kwargs=policy_kwargs, 
            learning_rate=linear_schedule(0.001), 
            seed=SEED,
            verbose=1)
        model_a2c.learn(total_timesteps=500000) 
        model_a2c.save("models/a2c_trading_agent")
        print("A2C Model saved!")

    # ---------------------------------------------------
    # PPO Training
    # ---------------------------------------------------
    if "ppo" in models_to_train:
        print("\n--- Training PPO Agent ---")
        model_ppo = PPO("MlpPolicy", env, 
            ent_coef=0.01, 
            policy_kwargs=policy_kwargs, 
            learning_rate=linear_schedule(0.001), 
            batch_size=128, 
            seed=SEED,
            verbose=1)
        model_ppo.learn(total_timesteps=500000)
        model_ppo.save("models/ppo_trading_agent")
        print("PPO Model saved!")

    # ---------------------------------------------------
    # DDPG Training
    # ---------------------------------------------------
    if "ddpg" in models_to_train:
        print("\n--- Training DDPG Agent ---")
        # DDPG needs action noise for exploration in continuous spaces
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        model_ddpg = DDPG("MlpPolicy", env, 
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(0.001),
            batch_size=128,
            seed=SEED,
            verbose=1)
        model_ddpg.learn(total_timesteps=500000)
        model_ddpg.save("models/ddpg_trading_agent")
        print("DDPG Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trading Agents")
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=['a2c', 'ppo', 'ddpg'], 
        default=['a2c', 'ppo', 'ddpg'],
        help="List of models to train (e.g., --models ddpg ppo). Defaults to all."
    )
    args = parser.parse_args()
    
    train_models(args.models)