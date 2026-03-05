import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

from load_data import fetch_and_preprocess_data
from trading_env import TradingEnv

def calculate_sharpe_ratio(net_worth_series, risk_free_rate=0.0):
    """Calculates the annualized Sharpe Ratio."""
    # Calculate daily returns: (Price_today - Price_yesterday) / Price_yesterday
    returns = pd.Series(net_worth_series).pct_change().dropna()
    if returns.std() == 0:
        return 0
    # Annualized Sharpe Ratio: (Mean Return - Risk Free Rate) / Std Dev * sqrt(252 trading days)
    sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    return sharpe

def evaluate_model(model, env, num_steps):
    """Runs a trained model through the environment and tracks net worth."""
    obs = env.reset()
    net_worths = []
    
    for _ in range(num_steps):
        # The model predicts the best action based on the observation
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        net_worths.append(info[0]['net_worth'])
        if done:
            break
            
    return net_worths

def test_models():
    print("Fetching testing data (Warming up indicators from Nov 2024, testing on 2025)...")
    # Fetch data starting slightly earlier so MACD/BBands can calculate without dropping 2025 dates
    raw_test_df = fetch_and_preprocess_data(ticker="SPY", start_date="2024-11-15", end_date="2026-01-01")
    
    # Strictly isolate the 2025 data by checking the Index
    test_df = raw_test_df[raw_test_df.index >= '2025-01-01'].reset_index()
    num_steps = len(test_df) - 1

    # Initialize the test environment
    test_env = DummyVecEnv([lambda: TradingEnv(df=test_df)])

    # NEW: Stack the last 5 days of observations so the agent sees momentum!
    test_env = VecFrameStack(test_env, n_stack=5)

    # 1. Load Trained Models
    model_a2c = A2C.load("models/a2c_trading_agent")
    model_ppo = PPO.load("models/ppo_trading_agent")

    # 2. Run Agents on 2025 Data
    print("\nRunning A2C Evaluation...")
    a2c_net_worth = evaluate_model(model_a2c, test_env, num_steps)
    
    print("Running PPO Evaluation...")
    ppo_net_worth = evaluate_model(model_ppo, test_env, num_steps)

    # 3. Calculate "Buy and Hold" Baseline
    print("Calculating Buy and Hold Baseline...")
    initial_balance = 10000
    first_day_price = test_df.loc[0, 'Close']
    shares_bought = initial_balance // first_day_price
    cash_leftover = initial_balance - (shares_bought * first_day_price)
    
    # Baseline net worth is the value of those shares every day + the leftover cash
    baseline_net_worth = [cash_leftover + (shares_bought * price) for price in test_df['Close'][:num_steps]]

    # 4. Calculate Metrics (Sharpe Ratio)
    a2c_sharpe = calculate_sharpe_ratio(a2c_net_worth)
    ppo_sharpe = calculate_sharpe_ratio(ppo_net_worth)
    baseline_sharpe = calculate_sharpe_ratio(baseline_net_worth)

    print("\n--- FINAL RESULTS (2025 Out-of-Sample) ---")
    print(f"Buy & Hold Final Value: ${baseline_net_worth[-1]:.2f} | Sharpe Ratio: {baseline_sharpe:.2f}")
    print(f"A2C Agent Final Value:  ${a2c_net_worth[-1]:.2f} | Sharpe Ratio: {a2c_sharpe:.2f}")
    print(f"PPO Agent Final Value:  ${ppo_net_worth[-1]:.2f} | Sharpe Ratio: {ppo_sharpe:.2f}")

    # 5. Plot Results
    dates = test_df['Date'][:num_steps]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, baseline_net_worth, label='Buy & Hold (Baseline)', linestyle='--', color='gray')
    plt.plot(dates, a2c_net_worth, label='A2C Agent', color='blue')
    plt.plot(dates, ppo_net_worth, label='PPO Agent', color='green')
    
    plt.title('Agent Performance vs. Market Baseline (2025)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Net Worth ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    test_models()