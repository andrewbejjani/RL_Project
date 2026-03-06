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

        self.avg_cost = 0.0  # <--- NEW: Tracks purchase price for the Stop-Loss module

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

        self.avg_cost = 0.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        
        max_price_scale = 1000.0 # SPY rarely exceeds 1000 in our timeframe
        max_shares_scale = 100.0 # Assuming we rarely hold more than 100 shares of SPY
        
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held / max_shares_scale,
            self.df.loc[self.current_step, 'Close'] / max_price_scale,
            self.df.loc[self.current_step, 'MACD'] / 10.0,
            self.df.loc[self.current_step, 'RSI'] / 100.0,
            self.df.loc[self.current_step, 'BB_Upper'] / max_price_scale,
            self.df.loc[self.current_step, 'BB_Lower'] / max_price_scale,
            self.df.loc[self.current_step, 'VIX'] / 100.0  
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
        
        # -------------------------------------------------------------------
        # PRO TRADER RL MODULE 1: Conviction Thresholds (Buy/Sell/Hold)
        # -------------------------------------------------------------------
        raw_action = action[0] 
        
        if raw_action > 0.7:
            target_percentage = 1.0  # High conviction: ALL IN
        elif raw_action < 0.3:
            target_percentage = 0.0  # High conviction: CASH OUT
        else:
            # Medium conviction: HOLD EXACT CURRENT POSITION (Zero fees paid)
            if current_portfolio_value > 0:
                target_percentage = (self.shares_held * current_price) / current_portfolio_value
            else:
                target_percentage = 0.0

        # -------------------------------------------------------------------
        # PRO TRADER RL MODULE 2: The Hardcoded Stop-Loss
        # -------------------------------------------------------------------
        # If we hold stock, check our unrealized return. 
        # If we are down 3% from our average purchase cost, OVERRIDE THE AI and FORCE A SELL.
        if self.shares_held > 0 and self.avg_cost > 0:
            unrealized_return = (current_price - self.avg_cost) / self.avg_cost
            if unrealized_return <= -0.03:  # <-- A strict 3% stop-loss
                target_percentage = 0.0     # Override AI, go to cash
                
        # Calculate target shares based on the final, filtered target_percentage
        target_stock_value = current_portfolio_value * target_percentage
        target_shares = int(target_stock_value / current_price)
        
        TRADING_FEE_RATE = 0.001 
        
        # Execute the rebalance
        if target_shares > self.shares_held:
            # Buying the difference
            shares_to_buy = target_shares - self.shares_held
            cost = shares_to_buy * current_price
            fee = cost * TRADING_FEE_RATE 
            
            if self.balance >= (cost + fee):  
                # NEW: Calculate the new average cost basis for our Stop-Loss module
                total_cost_basis = (self.shares_held * self.avg_cost) + cost
                self.avg_cost = total_cost_basis / target_shares
                
                self.balance -= (cost + fee)
                self.shares_held += shares_to_buy
                
        elif target_shares < self.shares_held:
            # Selling the difference
            shares_to_sell = self.shares_held - target_shares
            revenue = shares_to_sell * current_price
            fee = revenue * TRADING_FEE_RATE 
            
            self.balance += (revenue - fee)  
            self.shares_held -= shares_to_sell
            
            # NEW: If we sold everything, reset the average cost
            if self.shares_held == 0:
                self.avg_cost = 0.0
            
        # Update Net Worth AFTER trading
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # ---------------------------------------------------------------------------
        # REWARD FUNCTION: Logarithmic Alpha 
        
        # 1. Calculate Log Returns
        portfolio_log_return = 0
        if prev_net_worth > 0:
            portfolio_log_return = np.log(self.net_worth / prev_net_worth)
            
        prev_price = self.df.loc[self.current_step - 1, 'Close']
        market_log_return = 0
        if prev_price > 0:
            market_log_return = np.log(current_price / prev_price)
            
        # 2. Reward is Log Alpha (beating the market in log space)
        log_alpha = portfolio_log_return - market_log_return
        step_reward = log_alpha * 1500  # Keep the multiplier to help the network learn
        # ----------------------------------------------------------------------------
        
        # We can track info for our ultimate goal: Sharpe Ratio
        info = {
            'step': self.current_step,
            'net_worth': self.net_worth,
            'price': current_price
        }
        
        truncated = False # We don't have early stopping conditions yet
        
        return self._get_observation(), step_reward, terminated, truncated, info