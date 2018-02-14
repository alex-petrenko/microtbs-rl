import gym
import envs

from algorithms import dqn

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(experiment, env_id, max_num_episodes=1000000, fps=6):
    env = gym.make(env_id)
    env.seed(0)

    params = dqn.AgentDqn.Params(experiment).load()
    agent = dqn.AgentDqn(env, params)
    agent.initialize()

    episode_rewards = []
    for _ in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0

        while not done:
            env.process_events()
            env.render_with_analytics(agent.analyze_observation(obs))
            action = agent.best_action(obs)
            obs, rew, done, _ = env.step(action)
            episode_reward += rew
            env.clock.tick(fps)

        if env.should_quit():
            break
        else:
            env.process_events()
            env.render_with_analytics(agent.analyze_observation(obs))
            env.clock.tick(fps)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

    agent.finalize()
    env.close()
    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = 'MicroTbs-CollectWithTerrain-v1'
    experiment = get_experiment_name_env_id(env_id, 'dqn_v3_inception')
    return enjoy(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
