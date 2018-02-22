import gym

from algorithms import a2c
from algorithms.a2c.a2c_utils import *
from algorithms.common import run_policy_loop

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(experiment, env_id, max_num_episodes=1000000, fps=7):
    env = gym.make(env_id)
    env.seed(0)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    return run_policy_loop(agent, env, max_num_episodes, fps)


def main():
    init_logger(os.path.basename(__file__))
    env_id = CURRENT_ENV
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)
    return enjoy(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
