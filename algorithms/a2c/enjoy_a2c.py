import gym
import envs

from algorithms import a2c
from algorithms.common import run_policy_loop

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(experiment, env_id, max_num_episodes=1000000, fps=6):
    env = gym.make(env_id)
    env.seed(0)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    return run_policy_loop(agent, env, max_num_episodes, fps)


def main():
    init_logger(os.path.basename(__file__))
    env_id = 'MicroTbs-CollectPartiallyObservable-v3'
    experiment = get_experiment_name(env_id, 'a2c_v5')
    return enjoy(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
