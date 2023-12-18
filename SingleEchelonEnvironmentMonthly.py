import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple, deque 
np.random.seed(0)

#Environment Variables
BUFFER_SIZE = int(5*1e5)  #replay buffer size
BATCH_SIZE = 128      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters
LR = 1e-4            # learning rate
UPDATE_EVERY = 4      # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Environment():
    def __init__(self, demand_records):
        self.n_period = len(demand_records)
        self.current_period = 1
        self.month = 0
        self.inv_level = 2500
        self.inv_pos = 2500
        self.capacity = 10000
        self.holding_cost = 300
        self.unit_price = 3000
        self.fixed_order_cost = 250
        self.variable_order_cost = 50
        self.lead_time = 2
        self.order_arrival_list = []
        self.demand_list = demand_records
        self.state = np.array([self.inv_pos] + self.convert_month(self.month))
        self.state_list = []
        self.state_list.append(self.state)
        self.action_list = []
        self.reward_list = []
            
    def reset(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.inv_level = 25
        self.inv_pos = 25
        self.current_period = 1
        self.month = 0
        self.state = np.array([self.inv_pos] + self.convert_month(self.month))
        self.state_list.append(self.state)
        self.order_arrival_list = []
        return self.state
        
    def step(self, action):
        if action > 0:
            y = 1
            self.order_arrival_list.append([self.current_period+self.lead_time, action])
        else:
            y = 0
        if len(self.order_arrival_list) > 0:
            if self.current_period == self.order_arrival_list[0][0]:
                self.inv_level = min(self.capacity, self.inv_level + self.order_arrival_list[0][1])
                self.order_arrival_list.pop(0)  
        demand = self.demand_list[self.current_period-1]
        units_sold = demand if demand <= self.inv_level else self.inv_level
        reward = units_sold*self.unit_price-self.holding_cost*self.inv_level - y*self.fixed_order_cost -action*self.variable_order_cost    
        self.inv_level = max(0,self.inv_level-demand)
        self.inv_pos = self.inv_level
        if len(self.order_arrival_list) > 0:
            for i in range(len(self.order_arrival_list)):
                self.inv_pos += self.order_arrival_list[i][1]
        self.month = (self.month+1)%12
        self.state = np.array([self.inv_pos] +self.convert_month(self.month))
        self.current_period += 1
        self.state_list.append(self.state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        if self.current_period > self.n_period:
            terminate = True
        else: 
            terminate = False
        return self.state, reward, terminate, {}
    
    def convert_month(self,d):
        if d == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if d == 1:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if d == 2:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if d == 3:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        if d == 4:
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
        if d == 5:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        if d == 6:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        if d == 7:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        if d == 8:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        if d == 9:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if d == 10:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        if d == 11:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
