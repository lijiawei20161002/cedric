import gym
import numpy as np
import random
from ddos_gym.envs.defense import Defense
import pandas as pd
import matplotlib.pyplot as plt

# Initialize parameters
alpha = 0.7
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
train_episodes = 1000
max_rounds = 10
account_limit = 4
mode = 'cedric'
'''
# Environment setup
env = gym.make('ddos-v0', mode=mode)
graph = Defense()
base = account_limit

# Helper function to convert state tuple to a unique number
def tuple_to_num(state_n):
    num = 0
    for i in range(len(state_n)):
        num = num*base + state_n[i]
    if num >= base**len(env.observation_space.spaces):
        num = base**len(env.observation_space.spaces) - 1
    if num < 0:
        num = 0
    return int(num)

# Initialize Q-tables for each agent
Q = {}
QC = {}
for agent in graph.agents:
    Q[agent] = np.zeros((base**len(env.observation_space.spaces), env.action_space.n))
    QC[agent] = np.zeros((base**len(env.observation_space.spaces), account_limit))

# Training loop
rewards = []
for episode in range(train_episodes):
    state_n = env.reset()
    action_n = {}
    invest_n = np.zeros(len(graph.agents))
    total_training_rewards = [0] * len(list(graph.agents))
    total_training_rewards.extend(total_training_rewards)
    total_training_rewards.extend(total_training_rewards)

    for step in range(max_rounds):
        current_event = env.ddos[env.time]
        victim_agent = current_event[1][0]
        for agent in graph.agents:
            if random.uniform(0, 1) > epsilon:
                # Exploitation: choose the best-known action
                action_n[agent] = np.argmax(Q[agent][tuple_to_num(state_n), :])
            else:
                # Exploration: choose a random action
                action_n[agent] = env.action_space.sample()
            if state_n[agent] > 0:
                invest_n[agent] = np.argmax(QC[agent][tuple_to_num(state_n),:])
            else:
                invest_n[agent] = 0

        # Take action and observe reward and next state
        new_state_n, reward_n = env.step(invest_n, action_n)

        for agent in graph.agents:
            # Update Q-table for Q-learning agents
            Q[agent][tuple_to_num(state_n), action_n[agent]] += alpha * (reward_n[agent] + discount_factor * np.max(Q[agent][tuple_to_num(new_state_n), :]) - Q[agent][tuple_to_num(state_n), action_n[agent]])
            total_training_rewards[agent] += reward_n[agent]
            state_n[agent] = new_state_n[agent]

        # Decay epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        env.time += 1

    rewards.append(total_training_rewards)

    for step in range(max_rounds):
        current_event = env.ddos[env.time]
        victim_agent = current_event[1][0]
        # Initialize Tit-for-Tat cooperation status
        tit_for_tat_cooperation_status = {agent: False for agent in graph.agents}
        for agent in graph.agents:
            action_n[agent] = 1 if tit_for_tat_cooperation_status[agent] else 0
            if action_n[agent]:
                invest_n[agent] = state_n[agent] #random.randint(0, state_n[agent])
            else:
                invest_n[agent] = 0

        # Take action and observe reward and next state
        new_state_n, reward_n = env.step(invest_n, action_n)

        for agent in graph.agents:
            tit_for_tat_cooperation_status[agent] = action_n[victim_agent]
            total_training_rewards[agent] += reward_n[agent]
            state_n[agent] = new_state_n[agent]

    rewards.append(total_training_rewards)

    for step in range(max_rounds):
        current_event = env.ddos[env.time]
        victim_agent = current_event[1][0]
        for agent in graph.agents:
            # Implement Accumulating Credits strategy
            action_n[agent] = 1  # Always cooperate
            if state_n[agent] > 0:
                invest_n[agent] = state_n[agent] #random.randint(0, state_n[agent])
            else:
                invest_n[agent] = 0

        # Take action and observe reward and next state
        new_state_n, reward_n = env.step(invest_n, action_n)
    rewards.append(total_training_rewards)

# After training, save rewards to CSV
df_rewards = pd.DataFrame(rewards)
df_rewards.to_csv('output/strategy.csv', index=False)'''
df_rewards = pd.read_csv('output/strategy.csv')

strategy_rewards = {
    'q_learning': df_rewards.iloc[0:train_episodes].reset_index(drop=True).mean(axis=1),
    'tit_for_tat': df_rewards.iloc[train_episodes:2*train_episodes].reset_index(drop=True).mean(axis=1),
    'accumulate_credits': df_rewards.iloc[2*train_episodes:].reset_index(drop=True).mean(axis=1)
}

# Convert the strategy rewards to a DataFrame for rolling mean 
df_strategy_rewards = pd.DataFrame(strategy_rewards)
df_strategy_rewards.to_csv('output/strategy_rewards.csv', index=False)

# Calculate the rolling mean for smoothing the plot
df_rolling_strategy_rewards = df_strategy_rewards.rolling(window=20).mean()

plt.figure(figsize=(8, 6))

# Plot each strategy's average rewards using the rolling mean
plt.plot(df_rolling_strategy_rewards['q_learning'], label='Q-Learning')
plt.plot(df_rolling_strategy_rewards['tit_for_tat'], label='Tit-for-Tat')
plt.plot(df_rolling_strategy_rewards['accumulate_credits'], label='Accumulate Credits')

plt.title('Average Rewards per Strategy Group per Episode', fontsize=14)
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Average Rewards', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)

plt.savefig('output/average_rewards_per_strategy.png')  # Save the plot as a PNG
#plt.show()  # Optionally display the plot