import os
import gym
import sys
import imageio
import logging
import matplotlib.pyplot as plt

import envs
from envs import micro_tbs
from algorithms import a2c
from algorithms.a2c import a2c_utils

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def record(experiment, env_id, num_episodes=30, fps=6):
    env = gym.make(env_id)
    env.render_resolution = 400
    env.seed(2)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    agent.initialize()

    footage_dir = join(experiment_dir(experiment), '.footage')
    ensure_dir_exists(footage_dir)

    game_screens = []

    for episode_idx in range(num_episodes):
        logger.info('Episode #%d', episode_idx)

        # Make sure the generated environment is challenging enough for our agent (to make it interesting to watch).
        # Re-generate new worlds until the right conditions are met.
        while True:
            obs = env.reset()
            border_num_obstacles = env.world_size ** 2 - env.mode.play_area_size ** 2
            num_obstacles = sum(isinstance(t, micro_tbs.Obstacle) for t in env.terrain.flatten())
            min_obstacles_in_play_area = 7
            min_gold_piles = 4

            enough_obstacles = num_obstacles >= border_num_obstacles + min_obstacles_in_play_area
            enough_gold = env.num_gold_piles >= min_gold_piles
            env_is_interesting = enough_gold and enough_obstacles
            if env_is_interesting:
                break

        step = 0
        done = False
        while not done:
            game_screens.append(env.render(mode='rgb_array'))
            action = agent.best_action(obs)
            obs, _, done, _ = env.step(action)
            step += 1

        game_screens.append(env.render(mode='rgb_array'))

    agent.finalize()
    env.close()

    logger.info('Rendering gif...')

    gif_name = join(footage_dir, '{}.gif'.format(experiment))
    kwargs = {'duration': 1.0 / fps}
    imageio.mimsave(gif_name, game_screens, 'GIF', **kwargs)
    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = a2c_utils.CURRENT_ENV
    experiment = get_experiment_name(env_id, a2c_utils.CURRENT_EXPERIMENT)
    return record(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
