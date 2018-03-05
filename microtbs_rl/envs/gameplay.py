"""
Play a version of the environment with human controls.

"""


import gym
from microtbs_rl import envs

from microtbs_rl.utils.common_utils import *

logger = logging.getLogger(os.path.basename(__file__))


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))

    env = gym.make(envs.COLLECT_PARTIALLY_OBSERVABLE_LATEST)
    env.seed(0)

    fps = 30
    episode_rewards = []
    while not env.should_quit():
        env.reset()

        done = False
        episode_reward = 0.0
        while not done:
            action = env.process_events()
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            env.clock.tick(fps)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

        if not env.should_quit():
            # display the end position in the game for a couple of sec
            for _ in range(fps):
                env.render()
                env.clock.tick(fps)
                env.process_events()

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
