import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """A custom stock trading environment for Gymnasium"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index() # Don't drop it, so 'Date' becomes a column
        self.initial_balance = initial_balance
        
        # Continuous Action: A percentage from 0.0 (100% cash) to 1.0 (100% stock)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State Space (S): Features include Balance, Shares Held, and our market data
        # Market data: Close, MACD, RSI, BB_Upper, BB_Lower, VIX
        self.num_features = 2 + 6 # Balance, Shares + 6 data columns (VIX added)
        
        # We use a Box space to represent continuous values for our state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self):
        # NEURAL NETWORK SCALING FIX
        # We divide by sensible constants to squish all values between ~ -1.0 and 1.0
        
        max_price_scale = 1000.0 # SPY rarely exceeds 600 in our timeframe
        max_shares_scale = 100.0 # Assuming we rarely hold more than 100 shares of SPY
        
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held / max_shares_scale,
            self.df.loc[self.current_step, 'Close'] / max_price_scale,
            self.df.loc[self.current_step, 'MACD'] / 10.0,
            self.df.loc[self.current_step, 'RSI'] / 100.0,
            self.df.loc[self.current_step, 'BB_Upper'] / max_price_scale,
            self.df.loc[self.current_step, 'BB_Lower'] / max_price_scale,
            self.df.loc[self.current_step, 'VIX'] / 100.0  # <--- NEW VIX FEATURE
        ], dtype=np.float32)
        
        # Clip values just in case extreme market events cause spikes
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs

    def step(self, action):
        self.current_step += 1
        
        # Check if we reached the end of the dataset
        terminated = self.current_step >= len(self.df) - 1
        
        current_price = self.df.loc[self.current_step, 'Close']
        prev_net_worth = self.net_worth
        
        # Evaluate current net worth based on today's price BEFORE trading
        current_portfolio_value = self.balance + (self.shares_held * current_price)
        
        # --- NEW CONTINUOUS TRADING LOGIC ---
        # The agent outputs a target percentage of net worth to hold in the market
        # action is now an array, e.g., [0.6], meaning 60% in SPY, 40% in Cash
        target_percentage = np.clip(action[0], 0.0, 1.0)
        
        # Calculate how much monetary value should be in stocks
        target_stock_value = current_portfolio_value * target_percentage
        
        # Calculate how many shares that equates to (using int to avoid fractional shares)
        target_shares = int(target_stock_value / current_price)
        
        # Execute the rebalance
        if target_shares > self.shares_held:
            # Buying the difference
            shares_to_buy = target_shares - self.shares_held
            cost = shares_to_buy * current_price
            if self.balance >= cost:  # Double check we have the cash
                self.balance -= cost
                self.shares_held += shares_to_buy
                
        elif target_shares < self.shares_held:
            # Selling the difference
            shares_to_sell = self.shares_held - target_shares
            revenue = shares_to_sell * current_price
            self.balance += revenue
            self.shares_held -= shares_to_sell
            
        # Update Net Worth AFTER trading
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # ---------------------------------------------------------------------------
        # REWARD FUNCTION: Active Return (Beating the Market)
        
        # 1. Calculate the Agent's daily percentage return
        portfolio_return = 0
        if prev_net_worth > 0:
            portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth
            
        # 2. Calculate the Market's daily percentage return (Buy & Hold baseline)
        prev_price = self.df.loc[self.current_step - 1, 'Close']
        market_return = 0
        if prev_price > 0:
            market_return = (current_price - prev_price) / prev_price
            
        # 3. The Reward is the difference between the two (Alpha)
        step_reward = (portfolio_return - market_return) * 100
        # ----------------------------------------------------------------------------
        
        # We can track info for our ultimate goal: Sharpe Ratio
        info = {
            'step': self.current_step,
            'net_worth': self.net_worth,
            'price': current_price
        }
        
        truncated = False # We don't have early stopping conditions yet
        
        return self._get_observation(), step_reward, terminated, truncated, info