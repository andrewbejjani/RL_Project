import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from load_data import fetch_and_preprocess_data
from trading_env import TradingEnv

def calculate_sharpe_ratio(net_worth_series, risk_free_rate=0.0):
    returns = pd.Series(net_worth_series).pct_change().dropna()
    if returns.std() == 0:
        return 0
    sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    return sharpe

def evaluate_model(model, env, num_steps):
    obs = env.reset()
    net_worths = []
    
    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        net_worths.append(info[0]['net_worth'])
        if done:
            break
            
    return net_worths

def test_models(models_to_test):
    print("Fetching testing data (Warming up indicators from Nov 2024, testing on 2025)...")
    raw_test_df = fetch_and_preprocess_data(ticker="SPY", start_date="2024-11-15", end_date="2026-01-01")
    
    test_df = raw_test_df[raw_test_df.index >= '2025-01-01'].reset_index()
    num_steps = len(test_df) - 1

    test_env = DummyVecEnv([lambda: TradingEnv(df=test_df)])
    test_env = VecFrameStack(test_env, n_stack=10)

    # 1. Calculate "Buy and Hold" Baseline First
    print("Calculating Buy and Hold Baseline...")
    initial_balance = 10000
    first_day_price = test_df.loc[0, 'Close']
    shares_bought = initial_balance // first_day_price
    cash_leftover = initial_balance - (shares_bought * first_day_price)
    
    baseline_net_worth = [cash_leftover + (shares_bought * price) for price in test_df['Close'][:num_steps]]
    baseline_sharpe = calculate_sharpe_ratio(baseline_net_worth)

    # 2. Dynamic Model Loading & Evaluation
    results = {}
    
    if "a2c" in models_to_test:
        print("Running A2C Evaluation...")
        model_a2c = A2C.load("models/a2c_trading_agent")
        results["A2C"] = evaluate_model(model_a2c, test_env, num_steps)
        
    if "ppo" in models_to_test:
        print("Running PPO Evaluation...")
        model_ppo = PPO.load("models/ppo_trading_agent")
        results["PPO"] = evaluate_model(model_ppo, test_env, num_steps)
        
    if "ddpg" in models_to_test:
        print("Running DDPG Evaluation...")
        model_ddpg = DDPG.load("models/ddpg_trading_agent")
        results["DDPG"] = evaluate_model(model_ddpg, test_env, num_steps)

    # 3. Print Final Results
    print("\n--- FINAL RESULTS (2025 Out-of-Sample) ---")
    print(f"Buy & Hold Final Value: ${baseline_net_worth[-1]:.2f} | Sharpe Ratio: {baseline_sharpe:.2f}")
    
    for name, net_worth in results.items():
        sharpe = calculate_sharpe_ratio(net_worth)
        print(f"{name} Agent Final Value:  ${net_worth[-1]:.2f} | Sharpe Ratio: {sharpe:.2f}")

    # 4. Plot Results
    dates = test_df['Date'][:num_steps]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, baseline_net_worth, label='Buy & Hold (Baseline)', linestyle='--', color='gray')
    
    colors = {"A2C": "blue", "PPO": "green", "DDPG": "orange"}
    for name, net_worth in results.items():
        plt.plot(dates, net_worth, label=f'{name} Agent', color=colors[name])
    
    plt.title('Agent Performance vs. Market Baseline (2025)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Net Worth ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Trading Agents")
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=['a2c', 'ppo', 'ddpg'], 
        default=['a2c', 'ppo', 'ddpg'],
        help="List of models to test (e.g., --models ddpg ppo). Defaults to all."
    )
    args = parser.parse_args()
    
    test_models(args.models)