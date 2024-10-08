import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
from algorithms.ddpg.ddpg_actor import DDPGActor
import logging

logging.basicConfig(level=logging.DEBUG)

class PPOArgs:
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

class DDPGArgs:
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
        self.use_multivar = False
        self.use_gmm = False
        self.use_inter_kl = False

def _t2n(x):
    return x.detach().cpu().numpy()

def play_game():
    global episode_rewards, episode_opponent_rewards
    print("play a new game")
    obs = env.reset()
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        obs, rewards, dones, infos = env.step(actions)

        ego_rewards = rewards[:num_agents // 2, ...]
        opponent_rewards = rewards[num_agents//2:, ...]
        episode_rewards += ego_rewards.sum()
        episode_opponent_rewards += opponent_rewards.sum()

        if dones.all():
            print(infos)
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        # print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    print("episode_rewards", episode_rewards)
    print("episode_opponent_rewards", episode_opponent_rewards)
    if episode_rewards >= episode_opponent_rewards:
        print("我方赢")
        return 1
    else:
        print("敌方赢")
        return -1

def play_multiple_games(n):
    global episode_rewards, episode_opponent_rewards
    ego_wins = 0
    opponent_wins = 0
    for i in range(n):
        episode_rewards = 0
        episode_opponent_rewards = 0
        result = play_game()
        if result == 1:
            ego_wins += 1
        else:
            opponent_wins += 1
        print(f"Game {i + 1} finished. Result: {'我方赢' if result == 1 else '敌方赢'}")
    
    print(f"Total games: {n}")
    print(f"Ego wins: {ego_wins}")
    print(f"Opponent wins: {opponent_wins}")

i = 1
num_agents = 2
ego_policy_index = 0
enm_policy_index = 1
episode_rewards = 0
episode_opponent_rewards = 0
ego_run_dir = "E:\code\论文代码\aircombat\Aircombat_RL\scripts\results\SingleCombat\1v1\NoWeapon\SelfplayContinuousAction\ppo\RNN+VCHSE+CMA\run7"
enm_run_dir = "E:\code\论文代码\aircombat\Aircombat_RL\scripts\results\SingleCombat\1v1\NoWeapon\SelfplayContinuousAction\ppo\RNN+VCHSE+CMA\run7"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleCombatEnv("1v1/NoWeapon/SelfplayContinuousAction")
env.seed(i)
PPOargs = PPOArgs()
DDPGargs = DDPGArgs()
ego_policy = PPOActor(PPOargs, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(PPOargs, env.observation_space, env.action_space, device=torch.device("cuda"))
# enm_policy = DDPGActor(DDPGargs, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

play_multiple_games(3)  # 这里指定对弈10局
