## RL Trading bot

We have trained a trading bot based on 3 different algortihms (A2C, PPO, DDPG)
To be able to deploy this project, you will need to run the following steps:

#### Creating a virtual environment

- python3 -m venv venv
- ./venv/Scripts/activate

#### Install all needed libraries

- pip install -r requirements.txt

#### Train the models (optional as they already exist in model folder)

- python3 train_agents.py

This already trains all 3 models, if we want to specifically choose one we can run

- python3 train_agents.py --models a2c ppo ddpg

#### Test the models (get performance graph across 1 trading year + Sharpe Ratio levels)

- python3 test_agents.py 