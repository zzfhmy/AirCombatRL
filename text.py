import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from cmaes import CMA

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs = self.policy_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        # Convert memory to tensors
        states, actions, rewards, log_probs, next_states, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards
        rewards_mean = discounted_rewards.mean()
        rewards_std = discounted_rewards.std() + 1e-6
        discounted_rewards = (discounted_rewards - rewards_mean) / rewards_std

        # Compute policy loss and value loss
        for _ in range(10):  # Train for 10 epochs
            action_probs = self.policy_network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.squeeze())

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * discounted_rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def evaluate(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs = self.policy_network(state)
        return action_probs

    def interact_with_environment(self, env, num_steps=2048):
        state = env.reset()
        for _ in range(num_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            self.store_transition((state, action, reward, log_prob, next_state, done))
            state = next_state
            if done:
                state = env.reset()


class CMA_PPO:
    def __init__(self, state_dim, action_dim, sigma=0.5, population_size=50):
        self.ppo = PPO(state_dim, action_dim)
        self.sigma = sigma
        self.population_size = population_size
        initial_parameters = self.flatten_parameters(self.ppo.policy_network.state_dict())
        self.es = CMA(mean=initial_parameters, sigma=self.sigma, population_size=self.population_size)

    def flatten_parameters(self, parameters):
        return np.concatenate([p.flatten() for p in parameters.values()])

    def set_parameters(self, flat_parameters):
        param_dict = self.ppo.policy_network.state_dict()
        idx = 0
        for key in param_dict.keys():
            param_shape = param_dict[key].shape
            param_size = param_dict[key].numel()
            param_dict[key] = torch.tensor(flat_parameters[idx:idx + param_size].reshape(param_shape), dtype=torch.float32)
            idx += param_size
        self.ppo.policy_network.load_state_dict(param_dict)

    def evaluate_policy(self, env):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = self.ppo.policy_network(torch.tensor(state, dtype=torch.float32))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        return total_reward

class CMA_ES_PPO:
    def __init__(self, ppo, sigma=0.5, population_size=50):
        self.ppo = ppo
        self.sigma = sigma
        self.population_size = population_size
        self.es = CMA(mean=self.flatten_parameters(ppo.policy_network.state_dict()), sigma=self.sigma, population_size=self.population_size)

    def flatten_parameters(self, state_dict):
        params = []
        for key, value in state_dict.items():
            params.append(value.flatten())
        return torch.cat(params).numpy()

    def set_parameters(self, flat_parameters):
        state_dict = self.ppo.policy_network.state_dict()
        start = 0
        for key, value in state_dict.items():
            end = start + value.numel()
            state_dict[key] = torch.tensor(flat_parameters[start:end]).view(value.size())
            start = end
        self.ppo.policy_network.load_state_dict(state_dict)

    def get_policy_parameters(self, flat_parameters):
        state_dict = self.ppo.policy_network.state_dict()
        start = 0
        for key, value in state_dict.items():
            end = start + value.numel()
            state_dict[key] = torch.tensor(flat_parameters[start:end]).view(value.size())
            start = end
        return state_dict

    def evaluate_policy(self, env):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action, _ = self.ppo.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward

    def optimize(self, env, max_generations=100):
        for generation in range(max_generations):
            
            # 使用 CMA-ES 选择候选解
            solutions = []
            for _ in range(self.population_size):
                solution, _ = self.es.ask()
                solutions.append(solution)

            # 评估候选解，并使用 CMA-ES 更新参数
            rewards = []
            for solution in solutions:
                self.set_parameters(solution)
                total_reward = self.evaluate_policy(env)
                rewards.append(-total_reward)  # CMA-ES 使用负的奖励来最大化性能

            self.es.tell(solutions, rewards)
            
            # 使用最优的解更新 PPO 网络参数
            best_solution = solutions[np.argmax(rewards)]
            best_parameters = self.get_policy_parameters(best_solution)
            self.ppo.policy_network.load_state_dict(best_parameters)

            # 使用PPO对策略网络进行局部优化
            self.ppo.interact_with_environment(env)
            self.ppo.train()

            # 更新 CMA-ES 的初始参数为 PPO 训练后的参数
            trained_parameters = self.flatten_parameters(self.ppo.policy_network.state_dict())
            self.es = CMA(mean=trained_parameters, sigma=self.sigma, population_size=self.population_size)

# 使用示例
if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    cma_ppo = CMA_PPO(state_dim, action_dim)
    cma_ppo.optimize(env)
