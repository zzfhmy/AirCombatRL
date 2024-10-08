import torch
import numpy as np
from .ddpg_actor import DDPGActor
from .ddpg_q import DDPGq


class DDPGPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = DDPGActor(args, self.obs_space, self.act_space, self.device)
        self.actor_target = DDPGActor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = DDPGq(args, self.obs_space, self.act_space, self.device)
        self.critic_target = DDPGq(args, self.obs_space, self.act_space, self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize target networks with the same weights as the main networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        Q, rnn_states_critic = self.critic(obs, actions, rnn_states_critic, masks)
        return Q, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor
    
    def get_values(self, obs, actions, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        Q, _ = self.critic_target(obs, actions, rnn_states_critic, masks)
        return Q

    def prep_training(self):
        #target不做训练
        self.actor.train()
        self.actor_target.eval()
        self.critic.train()
        self.critic_target.eval()


    def prep_rollout(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def copy(self):
        return DDPGPolicy(self.args, self.obs_space, self.act_space, self.device)
