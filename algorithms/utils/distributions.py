import torch
import torch.nn as nn

from .utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# Standardize distribution interfaces


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        # Single: [1] => [] => [] => [1, 1] => [1] => [1]
        # Batch: [N]/[N, 1] => [N] => [N] => [N, 1] => [N] => [N, 1]
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.squeeze(-1).unsqueeze(-1).size())
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)
    


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return self.mean

# 多元Normal
class FixedMultivariateNormal(torch.distributions.MultivariateNormal):
    def log_probs(self, actions):
        return super().log_prob(actions).unsqueeze(-1)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

    def mode(self):
        return self.mean

# 混合同族分布
import torch.distributions as D
class FixedMixtureSameFamily(torch.distributions.MixtureSameFamily):
    def __init__(self, action_mean, logit_weights, covariance, validate_args=None):
        mixture_distribution = D.Categorical(logits=logit_weights)              # 分类分布
        component_distribution = D.MultivariateNormal(action_mean, covariance)  # 多元高斯分布
        super().__init__(mixture_distribution, component_distribution, validate_args)

    def log_probs(self, actions):
        return super().log_prob(actions).unsqueeze(-1)
    
    def entropy(self):
        # 手动计算GMM的熵
        gmm_entropy = (self._component_distribution.entropy() * self._mixture_distribution.probs).sum(-1)
        return gmm_entropy.unsqueeze(-1)
    
    def approx_inter_kl(self, actions):
        # 近似内部KL散度
        approx_inter_kl = torch.sum((super().log_prob(actions).unsqueeze(-1) - self._component_distribution.log_prob(actions.unsqueeze(1))) * self._mixture_distribution.probs, dim=-1)
        return approx_inter_kl.unsqueeze(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        # Single: [K] => [K] => [1]
        # Batch: [N, K] => [N, K] => [N, 1]
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Categorical, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.logits_net(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:
        return 1


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        # Initialize mean and log-variance layers
        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.mu_net(x)
        log_std = self.log_std_net(x)
        action_std = torch.exp(log_std)
        return FixedNormal(action_mean, action_std)
    
    def get_mean_var_logvar(self, x):
        action_mean = self.mu_net(x)
        log_std = self.log_std_net(x)
        return action_mean, torch.exp(log_std), log_std

    @property
    def output_size(self) -> int:
        return self._num_outputs
    
# 多元高斯分布
class MultivariateGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(MultivariateGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        self.covariance = CovarianceParameterization(num_outputs)  # 协方差矩阵
        self._num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.mu_net(x)
        return FixedMultivariateNormal(action_mean, self.covariance())

    def get_mean_var_logvar(self, x):
        action_mean = self.mu_net(x)
        return action_mean, self.covariance(), torch.log(self.covariance())

    @property
    def output_size(self) -> int:
        return self._num_outputs

# 参数化协方差矩阵
class CovarianceParameterization(nn.Module):
    def __init__(self, num_features):
        super(CovarianceParameterization, self).__init__()
        self.L = nn.Parameter(torch.zeros(num_features, num_features))
        self.log_diag = nn.Parameter(torch.zeros(num_features))

    def forward(self):
        # 创建下三角矩阵
        L = torch.tril(self.L)
        # 在对角线加正数来保证正定
        # diag_add = torch.exp(self.L.diagonal())
        diag_add = torch.exp(self.log_diag)
        covariance_matrix = torch.mm(L, L.t()) + torch.diag(diag_add) * 0.01
        return covariance_matrix
    
# 参数化对角协方差矩阵   
class DiagonalCovarianceParameterization(nn.Module):
    def __init__(self, num_features):
        super(DiagonalCovarianceParameterization, self).__init__()
        self.log_diag = nn.Parameter(torch.zeros(num_features))  # 对角线元素的对数值

    def forward(self):
        diag = torch.exp(self.log_diag)  # 确保对角线元素为正
        covariance_matrix = torch.diag(diag)
        return covariance_matrix, self.log_diag

# 多元高斯混合模型
class MultivariateGaussianMixtureModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01, num_components=4):
        super(MultivariateGaussianMixtureModel, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.mu_net = init_(nn.Linear(num_inputs, num_outputs * num_components + num_components))  # num_components组均值和num_components个weights
        self.covariance = nn.ModuleList([CovarianceParameterization(num_outputs) for _ in range(num_components)]) # num_components个协方差矩阵
        self._num_outputs = num_outputs
        self.num_components = num_components

    def forward(self, x):
        B = x.shape[0]
        action_mean = self.mu_net(x)
        action_mean, logit_weights = action_mean[:, :-self.num_components], action_mean[:, -self.num_components:]
        action_mean = action_mean.view(B, self.num_components, -1)  # 拆分成num_components份
        covariance = torch.cat([self.covariance[i]().unsqueeze(0) for i in range(self.num_components)], dim=0)
        return FixedMixtureSameFamily(action_mean, logit_weights, covariance)  # 高斯混合模型

    @property
    def output_size(self) -> int:
        return self._num_outputs

class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(BetaShootBernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, **kwargs):
        x = self.net(x)
        x = self.constraint(x) # contrain alpha, beta >=0
        x = 100 - self.constraint(100-x) # constrain alpha, beta <=100
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        alpha_0 = kwargs['alpha0']
        beta_0 = kwargs['beta0']
        # print(f"{alpha}, {beta}, {alpha_0}, {beta_0}")
        p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)
        return FixedBernoulli(p)

    @property
    def output_size(self) -> int:
        return self._num_outputs

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Bernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        x = self.logits_net(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:
        return self._num_outputs
