import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import random

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 64

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device).reshape(-1, 1)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).to(device).reshape(-1, 1)

        # Critic 更新
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * self.discount * target_q).detach()

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 更新
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

# 训练环境
env = gym.make('Pendulum-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.add_to_replay_buffer(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        agent.train()

        if done:
            break

    print(f"Episode: {episode}, Reward: {episode_reward}")

env.close()
