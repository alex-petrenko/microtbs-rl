import gym
import numpy as np

from baselines import deepq

from envs import MicroTbs

from utils.common_utils import *
from utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))


class OpenaiDqnMonitor(Monitor):
    def callback(self, local_vars, _):
        timestep = local_vars['t']
        if timestep % 10 == 0:
            mean_reward = np.mean(local_vars['episode_rewards'][-100:])
            self.progress_file.write('{},{}\n'.format(timestep, mean_reward))
            self.progress_file.flush()


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectWithTerrain-v0')
    env.seed(0)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 3, 1), (64, 3, 1)],
        hiddens=[256, 256],
        dueling=True,
    )

    experiment = get_experiment_name(env, 'openai_dqn')

    with OpenaiDqnMonitor(experiment) as monitor:
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=50000,
            buffer_size=100000,
            exploration_fraction=0.8,
            exploration_final_eps=0.0,
            print_freq=50,
            gamma=0.9,
            prioritized_replay=True,
            checkpoint_freq=1000,
            callback=monitor.callback,
        )

    filename = join(model_dir(experiment), experiment + '.pkl')
    logger.info('Saving model to %s', filename)
    act.save(filename)


if __name__ == '__main__':
    sys.exit(main())
