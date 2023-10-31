import os
from statistics import mean, median, stdev
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


# Normalize the training rewards within 0-1 range
def normalize_episode_reward(coeff_cont):
    coeff_cont = "{:.2f}".format(coeff_cont)
    coeff_util = "{:.2f}".format(1 - float(coeff_cont))
    deepshare_home = os.path.expandvars('$DEEPSHARE_PATH')
    log_path = f"{deepshare_home}/scheduler/training-log-{coeff_cont}-{coeff_util}"

    episode_rewards = []
    with open(log_path, 'r') as f:
        for line in f:
            if 'episode reward' in line:
                episode_rewards.append(float(line.split(" ")[7]))

    min_reward = -float(coeff_cont)
    max_reward = float(coeff_util)

    norm_episode_rewards = []
    for r in episode_rewards:
        norm_reward = (r - min_reward) / (max_reward - min_reward)
        _logger.debug(f'reward ({"{:.2f}".format(min_reward)}~{"{:.2f}".format(max_reward)}) {"{:.2f}".format(r)} norm (0-1) {"{:.2f}".format(norm_reward)}')
        norm_episode_rewards.append(norm_reward)

    _logger.debug(f'Extracted episode rewards from {log_path} (actual range: ({"{:.2f}".format(min_reward)}~{"{:.2f}".format(max_reward)}) normalized to 0~1): {norm_episode_rewards} len {len(norm_episode_rewards)}')

    return norm_episode_rewards

def draw_reward_curve_single(cont_util, norm_rewards):
    with open(f'../norm-episode-reward-{cont_util}.log', 'w') as f:
        episode = 0
        x_axis = []
        y_axis = []
        for r in norm_rewards:
            reward = "{:.2f}".format(r)
            _logger.debug(f'{episode} {reward}')
            x_axis.append(episode)
            y_axis.append(float(reward))
            episode += 1

        y_arr = np.array(y_axis)
        average = "{:.4f}".format(mean(y_axis))
        standard_dev = "{:.4f}".format(stdev(y_axis))
        med = "{:.4f}".format(median(y_axis))
        p90 = "{:.4f}".format(np.percentile(y_arr, 90)) # 90th percentile

        title = f'[Reward] Avg: {average} Stdev: {standard_dev} Med: {med} p90: {p90}'
        plt.plot(x_axis, y_axis)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Training Reward')
        plt.legend(f'{cont_util}')
        plt.savefig(f'norm-reward-curve-{cont_util}.pdf')


def draw_reward_curve_multiple(cont_util_to_reward_dict):
    title = f'Training Reward Curve'
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')

    for cont_util, norm_rewards in cont_util_to_reward_dict.items():
        with open(f'../norm-episode-reward-{cont_util}.log', 'w') as f:
            episode = 0
            x_axis = []
            y_axis = []
            for r in norm_rewards:
                reward = "{:.2f}".format(r)
                _logger.debug(f'{episode} {reward}')
                x_axis.append(episode)
                y_axis.append(float(reward))
                episode += 1

            plt.plot(x_axis, y_axis)

    plt.legend(['0.3', '0.4', '0.5', '0.6', '0.7'], loc='lower right', fancybox=True)

    plt.savefig(f'norm-reward-curve-{cont_util}.pdf')


def draw_reward_curve():
    cont_util_to_reward_dict = {}
    coeff_cont = 0.00
    for i in range(3, 10):
        coeff_cont = i*0.1
        norm_episode_rewards = normalize_episode_reward(coeff_cont)
        cont_util = f'{"{:.2f}".format(coeff_cont)}:{"{:.2f}".format(1-coeff_cont)}'
        cont_util_to_reward_dict[cont_util] = norm_episode_rewards
        _logger.debug(f'cont_util {cont_util} cont_util_to_reward_dict {len(cont_util_to_reward_dict[cont_util])}')

    draw_reward_curve_multiple(cont_util_to_reward_dict)

def main():
    draw_reward_curve()


if __name__ == "__main__":
    main()