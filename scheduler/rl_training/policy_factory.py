import glob
import time
import logging

from stable_baselines3 import PPO, A2C, DQN, TD3, SAC

_logger = logging.getLogger(__name__)

class PolicyFactory:
    def __init__(self):
        self.policy = None

    def save_checkpoint(self, ckpt_dir):
        # Checkpoint file will be saved and loaded as a .zip file by the SB3
        ckpt_file = f'{ckpt_dir}/{self.rl_algo}_{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}'
        _logger.debug(f'Saving checkpoint file {ckpt_file}')
        self.policy.save(ckpt_file)

    def load_checkpoint(self, ckpt_dir):
        # Load the most recent checkpoint of the RL algorithm
        ckpt_files = glob.glob(f'{ckpt_dir}/{self.rl_algo}*')
        if (len(ckpt_files) == 0):
            with open("ckpt_loading.log", "a+") as out:
                out.write(f'No checkpoint for {self.rl_algo} yet\n')
            return None
        else:
            ckpt_file = max(ckpt_files)
            with open("ckpt_loading.log", "a+") as out:
                out.write(f'Loading checkpoint {ckpt_file} (the most recent one among {ckpt_files})\n')
            return ckpt_file

    def _create_a2c_policy(self, env):
        self.policy = A2C("MlpPolicy", env=env, device='cpu', verbose=1)

    def _create_ppo_policy(self, env):
        self.policy = PPO("MlpPolicy", env=env, device='cpu', verbose=1)

    def _create_dqn_policy(self, env):
        self.policy = DQN("MlpPolicy", env=env, device='cpu', verbose=1)

    def _create_td3_policy(self, env):
        self.policy = TD3("MlpPolicy", env=env, device='cpu', verbose=1)

    def _create_sac_policy(self, env):
        self.policy = SAC("MlpPolicy", env=env, device='cpu', verbose=1)

    # Factory method that returns a requested RL algorithm-based policy
    def _create_policy(self, env, rl_algo='PPO'):
        if rl_algo == 'A2C':
            self._create_a2c_policy(env)
        elif rl_algo == 'PPO':
            self._create_ppo_policy(env)
        elif rl_algo == 'DQN':
            self._create_dqn_policy(env)
        elif rl_algo == 'TD3':
            self._create_td3_policy(env)
        elif rl_algo == 'SAC':
            self._create_sac_policy(env)
        else:
            raise ValueError(f'{rl_algo} is not supported')

    # A wrapper of the factory method that adds exception handling
    def create_policy(self, env, rl_algo):
        try:
            self._create_policy(env, rl_algo)
            return self.policy
        except ValueError as e:
            _logger.debug(e)
        return None
