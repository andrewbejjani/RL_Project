# 1. Import the data function from your first file
# (Make sure 'load_data.py' is in the same folder)
from load_data import fetch_and_preprocess_data 
from trading_env import TradingEnv # Only needed if you put this in a 3rd file

if __name__ == "__main__":
    print("1. Loading S&P 500 Data...")
    # Fetch the data using the function we built in Phase 1
    market_data = fetch_and_preprocess_data("SPY", "2015-01-01", "2025-01-01")
    
    print("\n2. Initializing the Trading Environment...")
    # Feed the data into our custom environment
    env = TradingEnv(df=market_data)
    
    # Reset the environment to the first day in the dataset
    obs, info = env.reset()
    
    print("\n3. Running a Random Agent Test for 5 Days...")
    for step in range(5):
        # The agent picks a completely random action (0: Hold, 1: Buy, 2: Sell)
        action = env.action_space.sample() 
        
        # The environment steps forward one day
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print what happened
        action_name = ["Hold", "Buy", "Sell"][action]

        current_date = env.df.loc[env.current_step, 'Date'].strftime('%Y-%m-%d')

        print(f"Date {current_date} | Action: {action_name:<4} | "
            f"Reward: ${reward:>7.2f} | Net Worth: ${info['net_worth']:>9.2f} | "
            f"SPY Price: ${info['price']:.2f}")
        
        if terminated:
            print("Reached the end of the dataset!")
            break