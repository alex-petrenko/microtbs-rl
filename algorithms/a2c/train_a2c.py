import gym
import numpy as np

import envs

from algorithms import a2c

from utils.common_utils import *
from utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))


class A2CMonitor(Monitor):
    def callback(self, local_vars, _):
        timestep = local_vars['step']
        if timestep % 10 == 0:
            self.progress_file.write('{},{}\n'.format(timestep, local_vars['avg_rewards']))
            self.progress_file.flush()


def main():
    init_logger(os.path.basename(__file__))

    env_id = 'MicroTbs-CollectPartiallyObservable-v2'
    experiment = get_experiment_name_env_id(env_id, 'a2c_v4')

    params = a2c.AgentA2C.Params(experiment)
    params.gamma = 0.925
    params.rollout = 10
    params.num_envs = 8
    params.train_for_steps = 1000000

    multithread_env = a2c.MultiEnv(params.num_envs, make_env_func=lambda: gym.make(env_id))

    agent = a2c.AgentA2C(multithread_env, params=params)
    agent.initialize()

    with A2CMonitor(experiment) as monitor:
        agent.learn(multithread_env, step_callback=monitor.callback)

    multithread_env.close()


if __name__ == '__main__':
    sys.exit(main())
