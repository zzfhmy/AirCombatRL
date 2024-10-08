import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
    
# Actor network
class SACActor(nn.Module):
    def __init__(self, obs_space, act_space):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, act_space)
        self.log_std = nn.Linear(256, act_space)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic network
class SACCritic(nn.Module):
    def __init__(self, obs_space, act_space):
        super(SACCritic, self).__init__()
        self.fc1 = nn.Linear(obs_space + act_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q
    

class SAC:
    def __init__(self, obs_space, act_space, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SACActor(obs_space, act_space).to(self.device)
        self.q1 = SACCritic(obs_space, act_space).to(self.device)
        self.q2 = SACCritic(obs_space, act_space).to(self.device)
        self.q1_target = SACCritic(obs_space, act_space).to(self.device)
        self.q2_target = SACCritic(obs_space, act_space).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args['lr'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=args['lr'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=args['lr'])

        self.log_alpha = torch.tensor(np.log(args['init_alpha']), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args['lr'])

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.target_entropy = -np.prod((act_space,)).item()
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        action = action.cpu().detach().numpy()[0]
        return action

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.log_alpha.exp() * next_log_prob
            q_target = reward + (1 - done) * self.gamma * min_q_next_target

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_action, new_log_prob = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * new_log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (new_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size, obs_space, act_space):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_space))
        self.action = np.zeros((max_size, act_space))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, obs_space))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs]
        )

class ReplayBuffer:
    def __init__(self, max_size, obs_space, act_space):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_space))
        self.action = np.zeros((max_size, act_space))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, obs_space))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs]
        )
    
args = {
    'lr': 3e-4,
    'init_alpha': 0.2,
    'gamma': 0.99,
    'tau': 0.005,
}

obs_space = 24  # Example observation space size
act_space = 4   # Example action space size
buffer_size = 1e6
batch_size = 256

sac = SAC(obs_space, act_space, args)
replay_buffer = ReplayBuffer(int(buffer_size), obs_space, act_space)

# Assume env is your environment
# state = env.reset()
# for each step in your environment loop
# action = sac.select_action(state)
# next_state, reward, done, _ = env.step(action)
# replay_buffer.add(state, action, reward, next_state, done)
# state = next_state
# if len(replay_buffer) > batch_size:
#     sac.update(replay_buffer, batch_size)
