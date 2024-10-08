import torch
import torch.nn as nn
import logging

from typing import Union, List
from .ddpg_policy import DDPGPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm

torch.autograd.set_detect_anomaly(True)

class DDPGTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args = args
        # DDPG config
        self.DDPG_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        #ddpg
        self.tau = 0.5
        self.gamma = args.gamma
        self.use_returns = False
    
    def DDPG_update(self, policy: DDPGPolicy, sample):
        obs_batch, actions_batch, next_obs_batch, masks_batch, next_masks_batch, rewards_batch, returns_batch, \
                value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch, next_rnn_states_actor_batch, next_rnn_states_critic_batch = sample

        rewards_batch = check(rewards_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        next_masks_batch = check(next_masks_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)

        with torch.no_grad():
            next_action, _, _ = policy.actor_target(next_obs_batch, next_rnn_states_actor_batch, next_masks_batch)
            q_next_target, _ = policy.critic_target(next_obs_batch, next_action, next_rnn_states_critic_batch, next_masks_batch)
            q_target = rewards_batch +  self.gamma * q_next_target * next_masks_batch

        q_target = returns_batch if self.use_returns == True else q_target
        current_q, _ = policy.critic(obs_batch, actions_batch, rnn_states_critic_batch, masks_batch)
        if self.use_clipped_value_loss:
            critic_pred_clipped = value_preds_batch + (current_q - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            critic_loss = (current_q - q_target).pow(2)
            critic_loss_clipped = (critic_pred_clipped - q_target).pow(2)
            critic_loss = 0.5 * torch.max(critic_loss_clipped, critic_loss)
        else:
            critic_loss = (current_q - q_target).pow(2)

        critic_loss = critic_loss.mean()  
        # Optimize the loss function
        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.critic_optimizer.step()

        #actor_loss = -self.critic(state, self.actor(state)).mean()
        action_new, _, _ = policy.actor(obs_batch, rnn_states_actor_batch, masks_batch)
        Q,_=policy.critic(obs_batch, action_new, rnn_states_critic_batch, masks_batch)
        actor_loss = -Q.mean()

        # Optimize the loss function
        policy.actor_optimizer.zero_grad()
        actor_loss .backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
        policy.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(policy.critic.parameters(), policy.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(policy.actor.parameters(), policy.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss

    def train(self, policy: DDPGPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['actor_loss'] = 0
        train_info['critic_loss'] = 0
        for _ in range(self.DDPG_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                # raise NotImplementedError
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:

                actor_loss, critic_loss = self.DDPG_update(policy, sample)
                train_info['actor_loss'] += actor_loss
                train_info['critic_loss'] += critic_loss


        num_updates = self.DDPG_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
