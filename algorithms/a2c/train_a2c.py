import gym
import numpy as np

from algorithms import a2c
from algorithms.a2c.a2c_utils import *

from utils.common_utils import *
from utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))


class A2CMonitor(Monitor):
    def callback(self, local_vars, _):
        timestep = local_vars['step']
        if timestep % 10 == 0:
            self.progress_file.write('{},{}\n'.format(timestep, local_vars['avg_rewards']))
            self.progress_file.flush()


def train(a2c_params, env_id):
    multithread_env = a2c.MultiEnv(a2c_params.num_envs, make_env_func=lambda: gym.make(env_id))

    agent = a2c.AgentA2C(multithread_env, params=a2c_params)
    agent.initialize()

    with A2CMonitor(a2c_params.experiment_name) as monitor:
        agent.learn(multithread_env, step_callback=monitor.callback)

    agent.finalize()
    multithread_env.close()
    return 0


def main():
    init_logger(os.path.basename(__file__))

    env_id = CURRENT_ENV
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)

    params = a2c.AgentA2C.Params(experiment)
    params.gamma = 0.95
    params.rollout = 10
    params.num_envs = 16
    params.train_for_steps = 20000
    return train(params, env_id)


if __name__ == '__main__':
    sys.exit(main())
