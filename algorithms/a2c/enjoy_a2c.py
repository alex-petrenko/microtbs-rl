import gym
import envs

from algorithms import a2c

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectPartiallyObservable-v2')
    env.seed(0)

    experiment = get_experiment_name(env, 'a2c_v4')
    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    agent.initialize()

    episode_rewards = []
    fps = 6
    while True:
        obs, done = env.reset(), False
        episode_reward = 0

        while not done:
            env.process_events()
            env.render()
            action = agent.best_action(obs)
            obs, rew, done, _ = env.step(action)
            episode_reward += rew
            env.clock.tick(fps)

        if env.should_quit():
            break
        else:
            env.process_events()
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
