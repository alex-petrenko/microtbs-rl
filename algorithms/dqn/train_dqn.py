import gym
import numpy as np

import envs
from algorithms import dqn

from utils.common_utils import *
from utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))


class DqnMonitor(Monitor):
    def callback(self, local_vars, _):
        timestep = local_vars['step']
        if timestep % 10 == 0:
            mean_reward = np.mean(local_vars['episode_rewards'][-100:])
            self.progress_file.write('{},{}\n'.format(timestep, mean_reward))
            self.progress_file.flush()


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectWithTerrain-v0')
    env.seed(0)

    experiment = get_experiment_name(env, 'dqn_v3_inception')
    params = dqn.AgentDqnSimple.Params(experiment)
    agent = dqn.AgentDqnSimple(env, params=params)
    agent.initialize()

    with DqnMonitor(experiment) as monitor:
        agent.learn(env, step_callback=monitor.callback)


if __name__ == '__main__':
    sys.exit(main())
