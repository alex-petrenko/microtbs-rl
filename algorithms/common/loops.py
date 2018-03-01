"""
Just a place to keep the code shared by many modules.

"""

import os
import logging


logger = logging.getLogger(os.path.basename(__file__))


def run_policy_loop(agent, env, max_num_episodes, fps):
    """Execute the policy and render onto the screen, using the standard agent interface."""
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
