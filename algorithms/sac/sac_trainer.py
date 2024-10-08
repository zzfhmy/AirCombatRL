import torch
import torch.nn as nn
import logging

from typing import Union, List
from .sac_policy import SACPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm

torch.autograd.set_detect_anomaly(True)

class SACTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args = args
        # SAC config
        self.SAC_epoch = args.ppo_epoch
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
        #sac
        self.tau = 0.005
        self.gamma = args.gamma
    
    def SAC_update(self, policy: SACPolicy, sample):

        obs_batch, actions_batch, next_obs_batch, masks_batch, next_masks_batch, rewards_batch, value_preds_batch,\
              rnn_states_actor_batch, rnn_states_critic_batch, next_rnn_states_actor_batch, next_rnn_states_critic_batch = sample

        next_obs_batch = check(next_obs_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        with torch.no_grad():
            next_action, next_log_prob, _ = policy.actor(next_obs_batch, next_rnn_states_actor_batch, next_masks_batch)
            q1_next_target, _ = policy.critic1_target(next_obs_batch, next_action, next_rnn_states_critic_batch, next_masks_batch)
            q2_next_target, _ = policy.critic2_target(next_obs_batch, next_action, next_rnn_states_critic_batch, next_masks_batch)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - policy.log_alpha.exp() * next_log_prob
            q_target = rewards_batch +  self.gamma * min_q_next_target

        current_q1,_ = policy.critic1(obs_batch, actions_batch, rnn_states_critic_batch, masks_batch)
        current_q2,_ = policy.critic2(obs_batch, actions_batch, rnn_states_critic_batch, masks_batch)
        if self.use_clipped_value_loss:
            critic1_pred_clipped = value_preds_batch + (current_q1 - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            critic2_pred_clipped = value_preds_batch + (current_q2 - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            critic1_loss = (current_q1 - q_target).pow(2)
            critic2_loss = (current_q2 - q_target).pow(2)
            critic1_loss_clipped = (critic1_pred_clipped - q_target).pow(2)
            critic2_loss_clipped = (critic2_pred_clipped - q_target).pow(2)
            critic1_loss = 0.5 * torch.max(critic1_loss_clipped, critic1_loss)
            critic2_loss = 0.5 * torch.max(critic2_loss_clipped, critic2_loss)
        else:
            critic1_loss = (current_q1 - q_target).pow(2)
            critic2_loss = (current_q2 - q_target).pow(2)
        
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()

        # Optimize the loss function
        policy.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.use_max_grad_norm:
            critic1_grad_norm = nn.utils.clip_grad_norm_(policy.critic1.parameters(), self.max_grad_norm).item()
        else:
            critic1_grad_norm = get_gard_norm(policy.critic1.parameters())
        policy.critic1_optimizer.step()

        policy.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.use_max_grad_norm:
            critic2_grad_norm = nn.utils.clip_grad_norm_(policy.critic2.parameters(), self.max_grad_norm).item()
        else:
            critic2_grad_norm = get_gard_norm(policy.critic2.parameters())
        policy.critic2_optimizer.step()


        action_new, new_log_probs, _ = policy.actor(obs_batch, rnn_states_actor_batch, masks_batch)
        q1_new ,_= policy.critic1(obs_batch, action_new, rnn_states_critic_batch, masks_batch)
        q2_new ,_= policy.critic2(obs_batch, action_new, rnn_states_critic_batch, masks_batch)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (policy.log_alpha.exp().detach() * new_log_probs - min_q_new).mean()

        # Optimize the loss function
        policy.actor_optimizer.zero_grad()
        actor_loss .backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
        policy.actor_optimizer.step()

        alpha_loss = -(policy.log_alpha.exp() * (new_log_probs + policy.target_entropy).detach()).mean()
        policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        policy.alpha_optimizer.step()

        for param, target_param in zip(policy.critic1.parameters(), policy.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(policy.critic2.parameters(), policy.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic1_loss, critic2_loss

    def train(self, policy: SACPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['actor_loss'] = 0
        train_info['critic1_loss'] = 0
        train_info['critic2_loss'] = 0
        for _ in range(self.SAC_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                # raise NotImplementedError
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:

                actor_loss, critic1_loss, critic2_loss = self.SAC_update(policy, sample)
                train_info['actor_loss'] += actor_loss
                train_info['critic1_loss'] += critic1_loss
                train_info['critic2_loss'] += critic2_loss

        num_updates = self.SAC_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
