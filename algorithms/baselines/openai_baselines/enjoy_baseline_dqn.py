import gym
import numpy as np
import tensorflow as tf

from baselines import deepq

import envs

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(experiment, env_id, max_num_episodes=1000000, fps=6):
    env = gym.make(env_id)
    env.seed(0)

    act = deepq.load(join(model_dir(experiment), experiment + '.pkl'))

    episode_rewards = []
    for _ in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0
        while not done:
            env.process_events()
            env.render()
            actions = act(obs[np.newaxis])
            action = actions[0]
            obs, rew, done, _ = env.step(action)
            episode_reward += rew
            env.clock.tick(fps)

        if env.should_quit():
            break

        env.process_events()
        env.render()
        env.clock.tick(fps)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = envs.COLLECT_WITH_TERRAIN_LATEST
    experiment = get_experiment_name(env_id, 'openai_dqn')
    return enjoy(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
