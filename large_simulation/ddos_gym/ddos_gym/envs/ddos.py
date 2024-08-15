import gym
from gym import spaces
import numpy as np
import csv
from ddos_gym.envs.defense import Defense
import random

init_balance = 4
samples = 5
account_limit = 4
max_agent_num = 200
random.seed(42)

class DDoS(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, mode='cedric', size=5):
        self.ddos = []
        self.time = 0
        self.mode = mode
        self.account_limit = account_limit
        self.max_agent_num = max_agent_num
        self.graph = Defense()
        self.ddos = self._load_ddos_data("ddos_gym/ddos_gym/envs/data/attack.csv")
        self.account = {agent: init_balance for agent in self.graph.agents}

        self.action_space = spaces.Discrete(2)
        self.account_space = [spaces.Discrete(self.account_limit)] * len(self.graph.agents)
        self.observation_space = spaces.Tuple([spaces.Discrete(max_agent_num)] + self.account_space)

    def _load_ddos_data(self, filepath):
        ddos_data = []
        with open(filepath, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                src = self._parse_countries(row[4])
                dst = self._parse_countries(row[1])
                bandwidth = float(row[3])
                ddos_data.append((src, dst, bandwidth))
        return ddos_data

    def _parse_countries(self, country_str):
        countries = []
        raw = country_str.split('[')[1].split(']')[0].split(",")
        for c in raw:
            if len(c.split("\'")) < 2:
                continue
            country_name = c.split("\'")[1].split("\'")[0]
            if country_name in self.graph.country_dict:
                countries.append(self.graph.country_dict[country_name])
        return countries

    def reset(self, seed=None, options=None):
        self.time = 0
        self.account = {agent: init_balance for agent in self.graph.agents}
        state = np.array(list(self.account.values()))
        return state

    def step(self, invest_n, action_n):
        reward_n = {}
        event = self.ddos[self.time]
        coalition = {agent for agent in self.graph.agents if action_n[agent] == 1}
        src, dst, bandwidth = event[0], event[1][0], event[2]
        success, gain = Defense(src, dst, coalition, bandwidth).social_gain()

        for agent in coalition:
            payoff, credit = self._calculate_payoff(agent, dst, gain, success, action_n, invest_n, coalition, src, bandwidth)
            self.account[agent] = int(self.account[agent] + credit/samples)
            reward_n[agent] = payoff

        if self.mode == 'shared':
            for agent in coalition:
                self.account[agent] += gain / len(coalition)

        state = np.array(list(self.account.values()))
        return state, reward_n

    def _calculate_payoff(self, agent, dst, gain, success, action_n, invest_n, coalition, src, bandwidth):
        payoff = 0
        credit = 0
        if dst == agent and success:
            payoff += self.graph.app[agent]
        if action_n[agent] == 1:
            payoff -= self.graph.cost[agent]
        payoff += gain

        if self.mode == 'cedric':
            credit = self._calculate_cedric_credit(agent, dst, coalition, src, bandwidth, invest_n, gain)
            payoff += credit / samples

        return payoff, credit

    def _calculate_cedric_credit(self, agent, dst, coalition, src, bandwidth, invest_n, gain):
        credit = 0
        iso = {agent}
        for _ in range(samples):
            subset = set(random.sample(coalition - iso, random.randint(1, len(coalition) - 1)) if len(coalition - iso) > 0 else coalition - iso)
            success1, g1 = Defense(src, dst, subset, bandwidth).social_gain()
            subset.add(agent)
            success2, g2 = Defense(src, dst, subset, bandwidth).social_gain()
            if dst == agent:
                credit -= invest_n[agent]
            if g2 > 0:
                credit += (g2 - g1) / g2 * invest_n[dst]
        return credit

    def render(self):
        if self.render_mode == "human":
            print(f'Step: {self.time}')

    def close(self):
        pass