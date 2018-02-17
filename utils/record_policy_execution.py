import os
import gym
import sys
import imageio
import logging
import matplotlib.pyplot as plt

import envs
from algorithms import a2c

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def record(experiment, env_id, num_episodes=1):
    env = gym.make(env_id)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    agent.initialize()

    footage_dir = join(experiment_dir(experiment), '.footage')
    ensure_dir_exists(footage_dir)

    for episode_idx in range(num_episodes):
        logger.info('Episode #%d', episode_idx)

        step = 0
        obs, done = env.reset(), False
        while not done:
            game_screen = env.render(mode='rgb_array')
            img_name = '{ep:05d}_{step:05d}.jpg'.format(ep=episode_idx, step=step)
            plt.imsave(join(footage_dir, img_name), game_screen)

            action = agent.best_action(obs)
            obs, _, done, _ = env.step(action)
            step += 1

    agent.finalize()
    env.close()

    logger.info('Rendering gif...')

    images = []
    image_files = sorted([f for f in os.listdir(footage_dir)])
    for image_fname in image_files:
        images.append(imageio.imread(join(footage_dir, image_fname)))
        os.unlink(join(footage_dir, image_fname))

    gif_name = join(footage_dir, '{}.gif'.format(experiment))
    kwargs = {'duration': 0.1}
    imageio.mimsave(gif_name, images, 'GIF', **kwargs)
    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = 'MicroTbs-CollectPartiallyObservable-v3'
    experiment = get_experiment_name(env_id, 'a2c_v5')
    return record(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
