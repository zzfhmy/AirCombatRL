import torch
import torch.nn as nn

from ..utils.mlp import MLPLayer, MLPqBase
from ..utils.gru import GRULayer
from ..utils.utils import check


class SACq(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(SACq, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPqBase(obs_space, act_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            print("使用RNN")
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) q module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.Q_out = nn.Linear(input_size, 1)
        self.to(device)

    def forward(self, obs, action, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        obs = torch.cat([obs, action], 1)
        critic_features = self.base(obs)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)
        Q = self.Q_out(critic_features)
        return Q, rnn_states
