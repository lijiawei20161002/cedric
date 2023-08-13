import gym
from gym import spaces
import numpy as np
import csv
from ddos_gym.envs.defense import Defense
import random

init_balance = 0
samples = 5
account_limit = 1

class DDoS(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, mode='cedric', size=5):
        self.ddos = []
        self.time = 0
        self.mode = mode
        self.account_limit = account_limit
        self.compression = 100
        self.graph = Defense()
        with open("ddos_gym/ddos_gym/envs/data/attack.csv", 'r') as f:
            csvreader = csv.reader(f) # delimiter='\t')
            next(csvreader)
            for row in csvreader:
                dst = []
                src = []
                rawsrc = row[4].split('[')[1].split(']')[0].split(",")
                for c in rawsrc:
                    if len(c) > 0:
                        cc = c.split("\'")[1].split("\'")[0]
                        if cc in self.graph.country_dict:
                            src.append(self.graph.country_dict[cc])
                rawdst = row[1].split('[')[1].split(']')[0].split(",")
                for d in rawdst:
                    if len(d) > 0:
                        cc = d.split("\'")[1].split("\'")[0]
                        if cc in self.graph.country_dict:
                            dst.append(self.graph.country_dict[cc])
                bandwidth = float(row[3])
                if len(src) == 0 or len(dst)==0:
                    continue
                self.ddos.append((src, dst, bandwidth))

        # We have 2 actions, corresponding to "join", "stay out"
        self.action_space = spaces.Discrete(2)
        self.account = [spaces.Discrete(self.account_limit)] * (len(self.graph.agents)+1)
        self.observation_space = spaces.Tuple([spaces.Discrete(self.account_limit)]+(self.account))
        for agent in self.graph.agents:
            self.account[agent] = init_balance

    def reset(self, seed=None, options=None):
        self.ddos = []
        self.time = 0
        self.graph = Defense()
        with open("ddos_gym/ddos_gym/envs/data/attack.csv", 'r') as f:
            csvreader = csv.reader(f) # delimiter='\t')
            next(csvreader)
            for row in csvreader:
                dst = []
                src = []
                rawsrc = row[4].split('[')[1].split(']')[0].split(",")
                for c in rawsrc:
                    if len(c) > 0:
                        cc = c.split("\'")[1].split("\'")[0]
                        if cc in self.graph.country_dict:
                            src.append(self.graph.country_dict[cc])
                rawdst = row[1].split('[')[1].split(']')[0].split(",")
                for d in rawdst:
                    if len(d) > 0:
                        cc = d.split("\'")[1].split("\'")[0]
                        if cc in self.graph.country_dict:
                            dst.append(self.graph.country_dict[cc])
                bandwidth = float(row[3])
                self.ddos.append((src, dst, bandwidth))
        for agent in self.graph.agents:
            self.account[agent] = init_balance
        return self.account

    def step(self, invest_n, action_n):
        reward_n = {}
        event = self.ddos[self.time]
        coalition = set()
        for agent in self.graph.agents:
            if action_n[agent] == 1:
                coalition.add(agent)
            reward_n[agent] = 0
        src = event[0]
        dst = event[1][0]
        bandwidth = event[2]
        success, gain = Defense(src, dst, coalition, bandwidth).social_gain()
        for agent in coalition:
            payoff = 0
            credit = 0
            self.account[len(self.account)-1] = invest_n[dst]
            if dst == agent:
                if success:
                    payoff += self.graph.app[agent]
            if action_n[agent] == 1:
                payoff -= self.graph.cost[agent]
            payoff += gain
            # cedric mode
            if self.mode == 'cedric':
                iso = set()
                iso.add(agent)
                for k in range(samples):
                    if len(coalition-iso)>0:
                        subset = random.sample(coalition-iso, random.randint(1, len(coalition)-1))
                        subset = set(subset)
                    else:
                        subset = coalition-iso
                    success1, g1 = Defense(src, dst, subset, bandwidth).social_gain()
                    subset.add(agent)
                    success2, g2 = Defense(src, dst, subset, bandwidth).social_gain()
                    subset.remove(agent)
                    if dst==agent:
                        credit -= invest_n[agent]
                    if action_n[agent] == 1:
                        if g2 > 0:
                            credit += (g2-g1)/g2*invest_n[dst]
                payoff += credit/samples
                self.account[agent] = int(self.account[agent] + credit/samples) // self.compression
            # counterfactual mode
            if self.mode == 'counterfactual':
                success1, g1 = Defense(src, dst, coalition, bandwidth).social_gain()
                success2, g2 = Defense(src, dst, coalition.remove(agent), bandwidth).social_gain()
                coalition.add(agent)
                self.account[agent] = int(self.account[agent] + (g2-g1)) // self.compression
            # no credit mode
            if self.mode == 'no credit':
                self.account[agent] = int(self.account[agent] + payoff) // self.compression
            reward_n[agent] = payoff
        # shared mode
        if self.mode == 'shared':
            for agent in coalition:
                self.account[agent] = int(self.account[agent] + gain/len(coalition)) // self.compression
        return self.account, reward_n

    def render(self):
        if self.render_mode == "human":
            print(f'Step: {self.current_step}')

    def _render_frame(self):
        return 0

    def close(self):
        return 0
