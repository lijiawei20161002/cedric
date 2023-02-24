import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from ddos_gym.envs.defense import Defense
import pandas as pd

env = gym.make('ddos-v0', )
graph = Defense()

alpha = 0.7
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 2000
#test_episodes = 100
max_rounds = 30
account_limit = 2
mode = 'cedric'  # mode can be 'cedric', 'no credit'

base = env.observation_space.spaces[0].n # observation space of each agent, should equal to account_limit

def tuple_to_num(state_n):
    num = 0
    for i in range(len(state_n)):
        num = num*base + state_n[i]
    if num >= base**len(env.observation_space.spaces):
        num = base**len(env.observation_space.spaces) - 1
    if num < 0:
        num = 0
    return num

Q = {}
QC = {}
for agent in graph.agents:
    Q[agent] = np.zeros((base**len(env.observation_space.spaces), env.action_space.n))
    QC[agent] = np.zeros((base**len(env.observation_space.spaces), account_limit))

states = []
actions = []
rewards = []

# the credits upper bound in an agent's account
env.account_limit = account_limit

for episode in range(train_episodes):
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
                action_n[agent] = np.argmax(Q[agent][tuple_to_num(state_n),:])
                if state_n[agent] > 0:
                    invest_n[agent] = np.argmax(QC[agent][tuple_to_num(state_n),:])
                else:
                    invest_n[agent] = 0
            else:
                action_n[agent] = env.action_space.sample()
                if state_n[agent] > 0:
                    invest_n[agent] = random.randint(0, state_n[agent])
                else:
                    invest_n[agent] = 0
            new_state_n, reward_n = env.step(invest_n, action_n)
            Q[agent][tuple_to_num(state_n), action_n[agent]] = Q[agent][tuple_to_num(state_n), action_n[agent]] + alpha*(reward_n[agent]+discount_factor*np.max(Q[agent][tuple_to_num(new_state_n), :])-Q[agent][state_n[agent], action_n[agent]])
            total_training_rewards[agent] += reward_n[agent]
            state_n[agent] = new_state_n[agent]

            epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
        env.time += 1
    print(episode)
    states.append(state_n)
    actions.append(action_n)
    rewards.append(total_training_rewards)
    

df1 = pd.DataFrame(states)
df2 = pd.DataFrame(actions)
df3 = pd.DataFrame(rewards)
df1.to_csv('output/states.csv', index=False)
df2.to_csv('output/actions.csv', index=False)
df3.to_csv('output/rewards.csv', index=False)