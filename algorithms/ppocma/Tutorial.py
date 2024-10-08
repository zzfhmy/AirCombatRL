class Policy:
    def __init__(self, state_dim, action_dim, action_min_limit, action_max_limit, mode="PPO-CMA",
                 entropy_loss_weight=0, network_depth=2, network_units=64, network_activation="lrelu",
                 network_skips=False, network_unit_norm_init=True, use_ppo_loss=False, separate_var_adapt=False,
                 learning_rate=0.001, min_sigma=0.01, use_sigma_soft_clip=True, ppo_epsilon=0.2, pi_epsilon=0,
                 n_history=1, global_variance=False, trainable_global_variance=True, use_gradient_clipping=False,
                 max_gradient_norm=0.5, negative_advantage_avoidance_sigma=0):
        self.network_depth = network_depth
        self.network_units = network_units
        self.network_activation = network_activation
        self.network_skips = network_skips
        self.network_unit_norm_init = network_unit_norm_init
        self.use_ppo_loss = use_ppo_loss
        self.separate_var_adapt = separate_var_adapt
        self.learning_rate = learning_rate
        self.min_sigma = min_sigma
        self.use_sigma_soft_clip = use_sigma_soft_clip
        self.ppo_epsilon = ppo_epsilon
        self.pi_epsilon = pi_epsilon
        self.n_history = n_history
        self.global_variance = global_variance
        self.trainable_global_variance = trainable_global_variance
        self.use_gradient_clipping = use_gradient_clipping
        self.max_gradient_norm = max_gradient_norm
        self.negative_advantage_avoidance_sigma = negative_advantage_avoidance_sigma

        max_sigma = 1.0 * (action_max_limit - action_min_limit)
        self.mode = mode

        # Some bookkeeping
        self.used_sigma_sum = 0
        self.used_sigma_sum_counter = 0

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min_limit = action_min_limit
        self.action_max_limit = action_max_limit

        # Define networks
        if state_dim == 0:
            self.policy_mean = nn.Parameter(torch.zeros(action_dim))
            self.policy_log_var = nn.Parameter(torch.log(torch.square(0.5 * (action_max_limit - action_min_limit)) * torch.ones(action_dim)))
            self.global_log_var_variable = self.policy_log_var
        else:
            if self.separate_var_adapt or self.global_variance:
                self.policy_mean = MLP(state_dim, action_dim, network_units, network_depth, network_activation)
                if self.global_variance:
                    self.policy_log_var = nn.Parameter(torch.log(torch.square(0.5 * (action_max_limit - action_min_limit)) * torch.ones(action_dim)))
                    self.global_log_var_variable = self.policy_log_var
                else:
                    self.policy_log_var = MLP(state_dim, action_dim, network_units, network_depth, network_activation)
            else:
                self.policy_mean_and_log_var = MLP(state_dim, action_dim * 2, network_units, network_depth, network_activation)
                self.policy_mean = lambda x: self.policy_mean_and_log_var(x)[:, :action_dim]
                self.policy_log_var = lambda x: self.policy_mean_and_log_var(x)[:, action_dim:]

        # Optimizers
        self.optimizer_mean = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer_var = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        if self.state_dim == 0:
            policy_mean = self.policy_mean
            policy_log_var = self.policy_log_var
        else:
            policy_mean = self.policy_mean(state)
            policy_log_var = self.policy_log_var(state)
        
        policy_mean = softClip(policy_mean, self.action_min_limit, self.action_max_limit)
        if self.use_sigma_soft_clip:
            max_log_var = torch.log(torch.square(max_sigma))
            min_log_var = torch.log(torch.square(self.min_sigma))
            policy_log_var = softClip(policy_log_var, min_log_var, max_log_var)
        policy_var = torch.exp(policy_log_var)
        policy_sigma = torch.sqrt(policy_var)
        return policy_mean, policy_sigma

    def loss(self, policy_mean, policy_var, policy_log_var, action_in, advantages_in, log_pi_old_in):
        if self.use_ppo_loss:
            log_pi = torch.sum(-0.5 * torch.square(action_in - policy_mean) / policy_var - 0.5 * policy_log_var, dim=1)
            if self.pi_epsilon == 0:
                r = torch.exp(log_pi - log_pi_old_in)
            else:
                r = torch.exp(log_pi) / (self.pi_epsilon + torch.exp(log_pi_old_in))
            per_sample_loss = torch.min(r * advantages_in, torch.clamp(r, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * advantages_in)
            policy_loss = -torch.mean(per_sample_loss)
            if entropy_loss_weight > 0:
                policy_loss -= entropy_loss_weight * 0.5 * torch.mean(torch.sum(policy_log_var, dim=1))
        else:
            policy_no_grad = policy_mean.detach()
            policy_var_no_grad = policy_var.detach()
            policy_log_var_no_grad = policy_log_var.detach()
            logp_no_mean_grad = -torch.sum(0.5 * torch.square(action_in - policy_no_grad) / policy_var + 0.5 * policy_log_var, dim=1)
            logp_no_var_grad = -torch.sum(0.5 * torch.square(action_in - policy_mean) / policy_var_no_grad + 0.5 * policy_log_var_no_grad, dim=1)
            pos_advantages = torch.nn.functional.relu(advantages_in)
            policy_sigma_loss = -torch.mean(pos_advantages * logp_no_mean_grad)
            policy_mean_loss = -torch.mean(pos_advantages * logp_no_var_grad)
            if self.negative_advantage_avoidance_sigma > 0:
                neg_advantages = torch.nn.functional.relu(-advantages_in)
                mirrored_action = old_policy_mean - (action_in - old_policy_mean)
                logp_no_var_grad_mirrored = -torch.sum(0.5 * torch.square(mirrored_action - policy_mean) / policy_var_no_grad + 0.5 * policy_log_var_no_grad, dim=1)
                effective_kernel_sq_width = self.negative_advantage_avoidance_sigma * self.negative_advantage_avoidance_sigma * policy_var_no_grad
                avoidance_kernel = torch.mean(torch.exp(-0.5 * torch.square(action_in - old_policy_mean) / effective_kernel_sq_width), dim=1)
                policy_mean_loss -= torch.mean((neg_advantages * avoidance_kernel) * logp_no_var_grad_mirrored)
            policy_loss = torch.mean(-advantages_in * logp_no_var_grad)
        return policy_loss

    def optimize_var(self, loss):
        self.optimizer_var.zero_grad()
        loss.backward()
        if self.use_gradient_clipping:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
        self.optimizer_var.step()

    def optimize_mean(self, loss):
        self.optimizer_mean.zero_grad()
        loss.backward()
        if self.use_gradient_clipping:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
        self.optimizer_mean.step()

    def init(self, state_mean, state_sd, action_mean, action_sd, n_minibatch=64, n_batches=4000, verbose=True):
        for batch_idx in range(n_batches):
            states = torch.normal(state_mean, state_sd, size=[n_minibatch, self.state_dim])
            actions = torch.normal(action_mean, action_sd, size=[n_minibatch, self.action_dim])
            policy_mean, policy_sigma = self.forward(states)
            init_loss = nn.MSELoss()(policy_mean, torch.tensor(action_mean)) + nn.MSELoss()(policy_sigma, torch.tensor(action_sd))
            self.optimize_var(init_loss)
            self.optimize_mean(init_loss)
            if verbose and (batch_idx % 100 == 0):
                print(f"Initializing policy with random Gaussian data, batch {batch_idx}/{n_batches}, loss {init_loss.item()}")

    def train(self, states, actions, advantages, n_minibatch, n_epochs, n_batches=0, state_offset=0, state_scale=1):
        history = deque()
        advantages_var = torch.var(advantages)
        for epoch in range(n_epochs):
            indices = np.arange(states.shape[0])
            np.random.shuffle(indices)
            for batch_idx in range(n_batches):
                batch_indices = indices[batch_idx*n_minibatch:(batch_idx+1)*n_minibatch]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                policy_mean, policy_sigma = self.forward(batch_states)
                loss_var = self.loss(policy_mean, policy_sigma, torch.log(policy_sigma**2), batch_actions, batch_advantages, None)
                self.optimize_var(loss_var)
                loss_mean = self.loss(policy_mean, policy_sigma, torch.log(policy_sigma**2), batch_actions, batch_advantages, None)
                self.optimize_mean(loss_mean)
                history.append((loss_var.item(), loss_mean.item()))
            if len(history) > 100:
                history.popleft()
            avg_loss_var = np.mean([loss[0] for loss in history])
            avg_loss_mean = np.mean([loss[1] for loss in history])
            print(f"Epoch {epoch+1}/{n_epochs}, Average Loss (Var): {avg_loss_var:.4f}, Average Loss (Mean): {avg_loss_mean:.4f}")

# Testing the code
if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    action_min_limit = torch.tensor([-1.0, -1.0])
    action_max_limit = torch.tensor([1.0, 1.0])
    policy = Policy(state_dim, action_dim, action_min_limit, action_max_limit)
    
    state_mean = torch.zeros(state_dim)
    state_sd = torch.ones(state_dim)
    action_mean = torch.zeros(action_dim)
    action_sd = torch.ones(action_dim) * 0.1

    policy.init(state_mean, state_sd, action_mean, action_sd)
    
    states = torch.normal(state_mean, state_sd, size=[1000, state_dim])
    actions = torch.normal(action_mean, action_sd, size=[1000, action_dim])
    advantages = torch.randn(1000)
    
    policy.train(states, actions, advantages, n_minibatch=64, n_epochs=10, n_batches=100)
