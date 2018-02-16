import gym
import numpy as np
import tensorflow as tf

from baselines import deepq

from envs import MicroTbs

from utils.common_utils import *
from utils.monitor import Monitor


logger = logging.getLogger(os.path.basename(__file__))

SESSION = None


class OpenaiDqnMonitor(Monitor):
    def callback(self, local_vars, _):
        # save the session handle and close the session before exiting
        global SESSION
        if SESSION is None:
            SESSION = local_vars['sess']

        timestep = local_vars['t']
        if timestep % 10 == 0:
            mean_reward = np.mean(local_vars['episode_rewards'][-100:])
            self.progress_file.write('{},{}\n'.format(timestep, mean_reward))
            self.progress_file.flush()


def train(experiment, env_id, train_for_steps=50000):
    env = gym.make(env_id)
    env.seed(0)

    tf.reset_default_graph()
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 3, 1), (64, 3, 1)],
        hiddens=[256, 256],
        dueling=True,
    )

    with OpenaiDqnMonitor(experiment) as monitor:
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=train_for_steps,
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

    # the "learn" function does not close the session before exit, hence the hack
    global SESSION
    if SESSION is not None:
        SESSION.__exit__(None, None, None)
        del SESSION

    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = 'MicroTbs-CollectWithTerrain-v1'
    experiment = get_experiment_name(env_id, 'openai_dqn')
    return train(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
