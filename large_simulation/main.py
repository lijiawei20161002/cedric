import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from ddos_gym.envs.defense import Defense
import pandas as pd

alpha = 0.7
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 2
#test_episodes = 100
max_rounds = 30
account_limit = 1
mode = 'cedric'  # mode can be 'cedric', 'no credit'
env = gym.make('ddos-v0', mode=mode)
graph = Defense()

base = account_limit 

def generate_states(table, string_number, base):  
    num_digits = len(string_number)  
    counter = [0] * num_digits  
  
    while True:  
        current_number = '[' + ','.join([str(x) for x in counter]) + ']'  
        table[current_number] = 0  
  
        # Increment the counter  
        index = 0  
        while index < num_digits:  
            counter[index] += 1  
            if counter[index] < base:  
                break  
            counter[index] = 0  
            index += 1  
  
        # If we've reached the maximum number, break the loop  
        if index == num_digits:  
            break  

Q = {}
QC = {}
for agent in graph.agents: 
    Q[agent] = {}
    generate_states(Q[agent], str(base) * len(env.observation_space.spaces), base)
    QC[agent] = {}
    generate_states(QC[agent], str(base) * len(env.observation_space.spaces), base)

states = []
actions = []
rewards = []

# the credits upper bound in an agent's account
env.account_limit = account_limit

for episode in range(train_episodes):
    print("Training episode: ", episode)
    state_n = env.reset()
    action_n = {}
    invest_n = np.zeros(len(graph.agents))
    total_training_rewards = {}
    for agent in graph.agents:
        total_training_rewards[agent] = 0
        action_n[agent] = 1

    for step in range(max_rounds):
        #agents = random.sample(graph.agents, 3)
        for agent in graph.agents:
            exp_exp_tradeoff = random.uniform(0,1)

            if exp_exp_tradeoff > epsilon:
                action_n[agent] = np.argmax(Q[agent][str(state_n)])
                if state_n[agent] > 0:
                    invest_n[agent] = np.argmax(QC[agent][str(state_n)])
                else:
                    invest_n[agent] = 0
            else:
                action_n[agent] = env.action_space.sample()
                if state_n[agent] > 0:
                    invest_n[agent] = random.randint(0, state_n[agent])
                else:
                    invest_n[agent] = 0
            new_state_n, reward_n = env.step(invest_n, action_n)
            new_state_n = [round(x) for x in new_state_n]
            print(new_state_n, reward_n)
            Q[agent][str(state_n)][action_n[agent]] = Q[agent][str(state_n)][action_n[agent]] + alpha*(reward_n[agent]+discount_factor*np.max(Q[agent][str(new_state_n)])-Q[agent][str(state_n)][action_n[agent]])
            total_training_rewards[agent] += reward_n[agent]
            state_n[agent] = new_state_n[agent]
            epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
        env.time += 1
    states.append(state_n)
    actions.append(action_n)
    rewards.append(total_training_rewards)
    df1 = pd.DataFrame(states)
    df1.to_csv('output/states_'+mode+'.csv', index=False)
    

#df1 = pd.DataFrame(states)
#df2 = pd.DataFrame(actions)
#df3 = pd.DataFrame(rewards)
#df1.to_csv('output/states_'+mode+'.csv', index=False)
#df2.to_csv('output/actions.csv', index=False)
#df3.to_csv('output/rewards.csv', index=False)