import gym
import numpy as np

import envs
from utils.common_utils import *

logger = logging.getLogger(os.path.basename(__file__))


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectSimple-v0')

    episode_rewards = []
    while not env.should_quit():
        obs, done = env.reset(), False
        episode_reward = 0
        while not done:
            env.process_events()
            env.render()
            obs, rew, done, _ = env.step(env.action_space.sample())
            episode_reward += rew

        env.render()

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )


if __name__ == '__main__':
    sys.exit(main())
