import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import matplotlib.pyplot as plt
import logging
import os
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
        self.use_multivar = True
        self.use_gmm = False
        self.use_inter_kl = False

def _t2n(x):
    return x.detach().cpu().numpy()

def multivariate_gaussian(pos, mu, Sigma):
    """生成多元高斯分布的概率密度值。
    pos: 样本位置 (x, y)
    mu: 均值向量
    Sigma: 协方差矩阵
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N

def gaosi_show(mu, Sigma, save_path=None):
    # 生成数据网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(x, y)

    # 将数据点放入矩阵中
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # 计算概率密度值
    z = multivariate_gaussian(pos, mu, Sigma)

    # 绘制3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    # 隐藏坐标轴
    ax.set_axis_off()

    if save_path:
        # 保存图片
        plt.savefig(save_path, dpi=600, pad_inches=0.0, bbox_inches='tight')
    plt.show()



i = 1
num_agents = 2
render = True
ego_policy_index = 10
enm_policy_index = 0
episode_rewards = 0
ego_run_dir = "/home/user/documents/zzf/aircombat/Aircambat-RL/scripts/results/SingleCombat/1v1/NoWeapon/SelfplayContinuousAction/ppo/ourppo/wandb/run-20240607_094842-ibe7rwz8/files"
enm_run_dir = "/home/user/documents/zzf/aircombat/Aircambat-RL/scripts/results/SingleCombat/1v1/NoWeapon/SelfplayContinuousAction/ppo/ourppo/wandb/run-20240607_094842-ibe7rwz8/files"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleCombatEnv("1v1/NoWeapon/SelfplayContinuousAction")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

# 动态生成保存路径
directory = 'gauss'
# 创建目录（如果不存在）
if not os.path.exists(directory):
    os.makedirs(directory)

print("Start render")
obs = env.reset()

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)

    # 打印高斯可视化
    ego_mean, ego_var, ego_logvar = ego_policy.get_meanvar(ego_obs, ego_rnn_states, masks, deterministic=True)
    
    # 将 Tensor 移动到 CPU 并转换为 NumPy 数组
    ego_mean = ego_mean.detach().cpu().numpy()
    ego_var = ego_var.detach().cpu().numpy()
    
    if(i<10):
        # 生成保存路径
        save_path = os.path.join(directory, f'gauss_{i}.jpg')
        gaosi_show(ego_mean, ego_var, save_path=save_path)

    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards

    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

    i += 1

print(episode_rewards)
