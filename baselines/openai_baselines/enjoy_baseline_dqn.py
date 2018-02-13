import gym
import numpy as np

from baselines import deepq

from envs import micro_tbs

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectWithTerrain-v0')
    env.seed(0)

    experiment = get_experiment_name(env, 'openai_dqn')
    act = deepq.load(join(model_dir(experiment), experiment + '.pkl'))

    episode_rewards = []
    fps = 6
    while not env.should_quit():
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

        env.render()
        env.clock.tick(fps)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )


if __name__ == '__main__':
    sys.exit(main())
