import torch
import numpy as np
from .sac_actor import SACActor
from .sac_q import SACq


class SACPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr
        self.init_alpha = 0.2

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = SACActor(args, self.obs_space, self.act_space, self.device)
        self.critic1 = SACq(args, self.obs_space, self.act_space, self.device)
        self.critic2 = SACq(args, self.obs_space, self.act_space, self.device)
        self.critic1_target = SACq(args, self.obs_space, self.act_space, self.device)
        self.critic2_target = SACq(args, self.obs_space, self.act_space, self.device)
        # Initialize target networks with the same weights as the main networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        self.log_alpha = torch.tensor(np.log(self.init_alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.target_entropy = -np.prod(self.act_space.shape).item()

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        Q1, rnn_states_critic1 = self.critic1(obs, actions, rnn_states_critic, masks)
        Q2, rnn_states_critic2 = self.critic2(obs, actions, rnn_states_critic, masks)
        values=torch.min(Q1, Q2)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic1

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic1_target.train()
        self.critic2_target.train()


    def prep_rollout(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

    def copy(self):
        return SACPolicy(self.args, self.obs_space, self.act_space, self.device)
