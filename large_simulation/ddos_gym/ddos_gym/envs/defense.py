import numpy as np
import csv
# app_value = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
app_value = [10] * 1000
# cost_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cost_value = [2] * 1000

class Defense:
    def __init__(self, src='US', dst='JP', coalition=['US', 'JP'], bandwidth=1):
        self.links = set()
        self.agents = set()
        self.country_dict = {}
        self.dict_country = {}
        self.app = {}
        self.cost = {}
        with open('ddos_gym/ddos_gym/envs/data/link.txt', 'r') as f:
            for line in f:
                a = line.split(',')[0].strip()
                b = line.split(',')[1].split('\n')[0].strip()
                if len(a)>0 and len(b)>0:
                    if a not in self.country_dict:
                        self.country_dict[a] = len(self.country_dict)
                        self.dict_country[self.country_dict[a]] = a
                        a = self.country_dict[a]
                        self.app[a] = app_value[a]
                        self.cost[a] = cost_value[a]
                    else:
                        a = self.country_dict[a]
                    if b not in self.country_dict:
                        self.country_dict[b] = len(self.country_dict)
                        self.dict_country[self.country_dict[b]] = b
                        b = self.country_dict[b]
                        self.app[b] = app_value[b]
                        self.cost[b] = cost_value[b]
                    else:
                        b = self.country_dict[b]
                    self.links.add(frozenset([a, b]))
                    self.agents.add(a)
                    self.agents.add(b)
        self.src = src
        self.dst = dst
        self.coalition = coalition
        self.bandwidth = bandwidth
        self.hop = 2
        #print(self.country_dict)
        #print(self.links)

    def set_src(self, src):
        self.src = src

    def set_dst(self, dst):
        self.dst = dst

    def set_coalition(self, coalition):
        self.coalition = coalition

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

    def defend(self, defense):
        rm=set()
        for l in self.links:
            if defense in l:
                rm.add(l)
        self.links = self.links - rm
                
    def path(self, frm, to):
        source = {frm}
        destination = {to}
        active = True
        cnt = 0
        while(active):
            active = False
            rm = set()
            for l in self.links:
                if source & l:
                    source = source.union(l)
                    rm.add(l)
                    active = True
                if destination & l:
                    destination = destination.union(l)
                    rm.add(l)
                    active = True
                if source & destination:
                    return True
            self.links = self.links - rm
            cnt = cnt + 1
            if cnt > self.hop:
                active = False
        return False

    def conduct(self):
        if self.coalition is not None:
            for c in self.coalition:
                self.defend(c)
            for c in self.src:
                if self.path(c, self.dst):
                    return True
        return False

    def social_gain(self):
        before = 0
        coalition = set()
        for agent in self.agents:
            for c in self.src:
                if (agent==c and (coalition is None or c not in coalition)) or (self.path(c, agent) and self.path(agent, self.dst)):
                    before = before + self.bandwidth
        #print(self.coalition)
        success = self.conduct()
        after = 0
        coalition = self.coalition
        for agent in self.agents:
            for c in self.src:
                if (agent==c and (coalition is None or c not in coalition)) or (self.path(c, agent) and self.path(agent, self.dst)):
                    after = after + self.bandwidth
        #print('gain', before, after)
        return (success, before - after) 

    
