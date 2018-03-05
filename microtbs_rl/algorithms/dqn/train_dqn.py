"""
Learn a policy with DQN algorithm.

"""


import gym
import numpy as np

from microtbs_rl import envs
from microtbs_rl.algorithms import dqn

from microtbs_rl.utils.common_utils import *
from microtbs_rl.utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))


class DqnMonitor(Monitor):
    def callback(self, local_vars, _):
        timestep = local_vars['step']
        if timestep % 10 == 0:
            mean_reward = np.mean(local_vars['episode_rewards'][-100:])
            self.progress_file.write('{},{}\n'.format(timestep, mean_reward))
            self.progress_file.flush()


def train(dqn_params, env_id):
    env = gym.make(env_id)
    env.seed(0)

    agent = dqn.AgentDqn(env, params=dqn_params)
    agent.initialize()

    with DqnMonitor(dqn_params.experiment_name) as monitor:
        agent.learn(env, step_callback=monitor.callback)

    agent.finalize()
    return 0


def main():
    init_logger(os.path.basename(__file__))

    env_id = envs.COLLECT_WITH_TERRAIN_LATEST
    experiment = get_experiment_name(env_id, 'dqn_v3_inception')
    params = dqn.AgentDqn.Params(experiment)
    return train(params, env_id)


if __name__ == '__main__':
    sys.exit(main())
