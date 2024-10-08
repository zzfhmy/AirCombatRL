import logging
import numpy as np
import torch

from algorithms.ppo.ppo_policy import PPOPolicy as Policy
from algorithms.utils.buffer import ReplayBuffer
from cmaes import CMA


def _t2n(x):
    return x.detach().cpu().numpy()

class CMAES:
    def __init__(self, eval_envs, all_args, device, sigma=0.5, population_size=10):
        self.all_args = all_args
        self.envs = eval_envs  # 请确保在实例化该类后设置 envs
        self.device = device
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        # 设置一些必要的属性
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.eval_episodes = self.all_args.eval_episodes
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents

        self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)
        self.opponent_ppo = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.ppo = Policy(self.all_args, self.obs_space, self.act_space, device=self.device) 

        self.sigma = sigma
        self.population_size = population_size

    def flatten_parameters(self, state_dict):
        params = []
        for k, v in state_dict.items():
            params.append(v.view(-1))
        return torch.cat(params)
    
    def parameters_to_state_dict(self, parameters, state_dict):
        idx = 0
        new_state_dict = {}
        for k, v in state_dict.items():
            num_params = v.numel()
            new_state_dict[k] = torch.tensor(parameters[idx:idx + num_params]).view(v.shape)
            idx += num_params
        return new_state_dict

    def load_solution_into_opponent(self, solution):
        new_state_dict = self.parameters_to_state_dict(solution, self.opponent_ppo.actor.act.state_dict())
        self.opponent_ppo.actor.act.load_state_dict(new_state_dict)

    def load_best_params_into_ppo(self, best_params):
        state_dict = self.ppo.actor.act.state_dict()
        new_state_dict = self.parameters_to_state_dict(best_params, state_dict)
        self.ppo.actor.act.load_state_dict(new_state_dict)

    def ask(self):
        return [self.es.ask() for _ in range(self.es.population_size)]
    
    def tell(self, solutions, fitness):
        self.es.tell(solutions, fitness)
    
    def result(self):
        return self.es.best
    
    def train_with_cma_es(self, ppo, num_generations=10):
        self.ppo = ppo
        trained_parameters = self.flatten_parameters(self.ppo.actor.act.state_dict()).cpu()  # 将张量从 GPU 移动到 CPU
        self.es = CMA(mean=trained_parameters.numpy(), sigma=self.sigma, population_size=self.population_size)
        
        for generation in range(num_generations):
            solutions = self.ask()
            rewards = []

            for solution in solutions:
                self.load_solution_into_opponent(solution)
                reward = self.eval(self.ppo, self.opponent_ppo)
                rewards.append(reward)
            
            self.tell(solutions, -np.array(rewards))

        best_params = self.result().x
        self.load_best_params_into_ppo(best_params)
        print("Best policy parameters found by CMA-ES")

    @torch.no_grad()
    def eval(self, ppo, opponent_ppo):
        logging.info("\nStart cma...")
        #调用了策略的准备方法，为接下来的 rollout 做准备。通常这意味着将策略设置为评估或执行状态。
        ppo.prep_rollout()
        opponent_ppo.prep_rollout()

        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards= np.zeros_like(cumulative_rewards)

        eval_each_episodes = self.eval_episodes // self.num_opponents

        eval_cur_opponent_idx = 0
        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                eval_cur_opponent_idx += 1

                # reset obs/rnn/mask
                obs = self.envs.reset()
                masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            # [Selfplay] get actions
            actions, rnn_states = ppo.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states \
                = opponent_ppo.act(np.concatenate(opponent_obs),
                                                np.concatenate(opponent_rnn_states),
                                                np.concatenate(opponent_masks), deterministic=True)
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            # Obser reward and next obs
            obs, eval_rewards, dones, eval_infos = self.envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(dones_env)

            # [Selfplay] Reset obs, masks, rnn_states
            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]), dtype=np.float32)

            opponent_masks[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)

            # [Selfplay] Get rewards
            opponent_rewards = eval_rewards[:, self.num_agents//2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env == True])
            cumulative_rewards[dones_env == True] = 0

        # Compute average episode rewards
        episode_rewards = np.concatenate(episode_rewards) # shape (self.eval_episodes, self.num_agents, 1)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1) # shape: (self.eval_episodes,)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1) # shape (self.num_opponents,)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)

        return  opponent_average_episode_rewards - eval_average_episode_rewards
    