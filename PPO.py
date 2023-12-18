import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the PPO agent
class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, clip_ratio):
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_probs[:, action.item()].item()

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages_cumulative = 0
        next_value = next_value if not dones[-1] else 0

        for t in reversed(range(len(rewards))):
            advantages_cumulative = advantages_cumulative * self.gamma + rewards[t] + next_value - values[t]
            next_value = values[t]
            advantages[t] = advantages_cumulative
            returns[t] = advantages_cumulative + values[t]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns

    def update_policy(self, states, actions, old_probs, advantages, returns):
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_probs = torch.FloatTensor(old_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Calculate the ratio of new and old policy probabilities
        new_probs = self.actor(states).gather(1, actions.unsqueeze(1))
        ratio = new_probs / old_probs

        # Calculate the surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (critic)
        critic_loss = F.mse_loss(self.critic(states).squeeze(1), returns)

        # Update actor and critic networks
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

def train(env, agent, num_episodes):
    episodic_reward = []
    episodic_actions = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        states, actions, old_probs, rewards, values = [], [], [], [], []

        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            old_probs.append(action_prob)
            rewards.append(reward*5*episode**0.5)
            values.append(agent.critic(torch.FloatTensor(state).to(device)).item())

            state = next_state
            total_reward += reward

        next_value = agent.critic(torch.FloatTensor(next_state).to(device)).item()
        advantages, returns = agent.compute_advantages(rewards, values, next_value, [done])

        agent.update_policy(states, actions, old_probs, advantages, returns)
        episodic_actions.append(actions)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
        episodic_reward.append(rewards)

    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, 'ppo.pt')
    return episodic_reward, episodic_actions