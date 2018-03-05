"""
Always handy to have a random agent.
If an RL agent performs significantly better than random, then it must be at least learning something.

"""


import gym

from microtbs_rl import envs

from microtbs_rl.algorithms.common import run_policy_loop
from microtbs_rl.algorithms.common.agent import AgentRandom

from microtbs_rl.utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(env_id, max_num_episodes=1000000, fps=6):
    env = gym.make(env_id)
    env.seed(0)
    agent = AgentRandom(params=None, env=env)
    return run_policy_loop(agent, env, max_num_episodes, fps)


def main():
    init_logger(os.path.basename(__file__))
    env_id = envs.COLLECT_WITH_TERRAIN_LATEST
    return enjoy(env_id)


if __name__ == '__main__':
    sys.exit(main())
