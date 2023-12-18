import torch.nn as nn
import torch
import numpy as np


BUFFER_SIZE = int(5*1e5)  #replay buffer size
BATCH_SIZE = 128      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters
LR = 1e-4            # learning rate
UPDATE_EVERY = 4      # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)

    
actor = Actor(12, 42)
critic = Critic(12)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
memory = Memory()
max_steps = 200

def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))

    # target values are calculated backward
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + GAMMA*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning

    advantage = torch.Tensor(q_vals) - values

    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()

    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()


def t(x): return torch.from_numpy(x).float()


def A2C_training(env, num_episodes):
  A2C_episode_rewards = []
  A2C_action_list = []
  for i in range(num_episodes):
    done = False
    total_reward = 0
    state = env.reset()
    steps = 0
    while not done:
      probs = actor(t(state))
      dist = torch.distributions.Categorical(probs=probs)
      action = dist.sample()
      next_state, reward, done, info = env.step(action.detach().data.numpy())

      total_reward += reward
      steps += 1
      memory.add(dist.log_prob(action), critic(t(state)), reward, done)
      state = next_state

      if done or (steps % max_steps == 0):
        last_q_val = critic(t(next_state)).detach().data.numpy()
        train(memory, last_q_val)
        memory.clear()
        if(i % 20 == 0):
          print(f'Episode Number: {i} Reward: {reward}')
    A2C_episode_rewards.append(total_reward)
    A2C_action_list.append(action)
  return A2C_episode_rewards, A2C_action_list