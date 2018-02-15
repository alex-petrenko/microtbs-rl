import json
import numpy as np

from utils.dnn_utils import *
from utils.common_utils import *

from algorithms.common import AgentLearner


logger = logging.getLogger(os.path.basename(__file__))


class Policy:
    class CategoricalProbabilityDistribution:
        """Based on https://github.com/openai/baselines implementation."""
        def __init__(self, logits):
            self.logits = logits

        def entropy(self):
            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        def sample(self):
            u = tf.random_uniform(tf.shape(self.logits))
            return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def __init__(self, input_shape, num_actions):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

        # process observations
        input_shape = [None] + input_shape  # add batch dimension
        self.observations = tf.placeholder(tf.float32, shape=input_shape)

        # convolutions
        # conv_filters = self._convnet_simple([(32, 3, 1), (64, 3, 1), (64, 3, 1)])
        conv_filters = self._inception()
        conv_out = tf.contrib.layers.flatten(conv_filters)

        # fully-connected layers to generate actions
        actions_fc = dense(conv_out, 256, self.regularizer)
        self.actions = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.actions_prob_distribution = Policy.CategoricalProbabilityDistribution(self.actions)
        self.act = self.actions_prob_distribution.sample()

        value_fc = dense(conv_out, 256, self.regularizer)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

        # summaries
        with tf.variable_scope('conv1', reuse=True):
            weights = tf.get_variable('weights')
            tf.summary.image('conv1/kernels', put_kernels_on_grid(weights), max_outputs=1)
        tf.summary.scalar('value', tf.reduce_mean(self.value))
        logger.info('Total parameters in the model: %d', count_total_parameters())

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self.regularizer, scope=scope)

    def _convnet_simple(self, convs):
        layer = self.observations
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer

    def _inception(self):
        conv_input = self._convnet_simple([(128, 3, 1)])

        with tf.variable_scope('branch1x1'):
            branch1x1 = self._conv(conv_input, 64, 1, 1)
        with tf.variable_scope('branch5x5'):
            branch5x5 = self._conv(conv_input, 48, 1, 1)
            branch5x5 = self._conv(branch5x5, 64, 5, 1)
        with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = self._conv(conv_input, 64, 1, 1)
            branch3x3dbl = self._conv(branch3x3dbl, 96, 3, 1)
            branch3x3dbl = self._conv(branch3x3dbl, 96, 3, 1)
        with tf.variable_scope('branch_pool'):
            branch_pool = avg_pool(conv_input, 3, 1)
            branch_pool = self._conv(branch_pool, 32, 1, 1)

        return tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])


class AgentA2C(AgentLearner):
    class Params(AgentLearner.Params):
        def __init__(self, experiment_name):
            super(AgentA2C.Params, self).__init__(experiment_name)

            self.gamma = 0.9  # future reward discount
            self.learning_rate = 1e-3
            self.rollout = 5  # number of successive env steps used for each model update
            self.num_envs = 16  # number of environments running in parallel. Batch size = rollout * num_envs

            # components of the loss function
            self.entropy_loss_coeff = 0.01
            self.value_loss_coeff = 0.5

            # training process
            self.save_every = 5000
            self.summaries_every = 100
            self.print_every = 50
            self.train_for_steps = 50000

        def _params_file(self):
            return join(experiment_dir(self.experiment_name), 'params.json')

        def ensure_serialized(self):
            if not self._params_serialized:
                self.serialize()
                self._params_serialized = True

        def serialize(self):
            with open(self._params_file(), 'w') as json_file:
                json.dump(self.__dict__, json_file, indent=2)

        def load(self):
            with open(self._params_file()) as json_file:
                self.__dict__ = json.load(json_file)
                return self

    def __init__(self, env, params):
        super(AgentA2C, self).__init__(params)

        input_shape = list(env.observation_space.shape)
        num_actions = env.action_space.n
        self.policy = Policy(input_shape, num_actions)

        self.selected_actions = tf.placeholder(tf.int32, [None])  # action selected by the policy
        self.value_estimates = tf.placeholder(tf.float32, [None])
        self.discounted_rewards = tf.placeholder(tf.float32, [None])  # estimate of total reward (rollout + value)

        advantages = self.discounted_rewards - self.value_estimates

        # negative logarithm of the probabilities of actions
        neglogp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.policy.actions, labels=self.selected_actions,
        )

        # maximize probabilities of actions that give high advantage
        action_loss = tf.reduce_mean(advantages * neglogp_actions)

        # penalize for inaccurate value estimation
        value_loss = tf.losses.mean_squared_error(self.discounted_rewards, self.policy.value)
        value_loss = self.params.value_loss_coeff * value_loss

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_loss = -tf.reduce_mean(self.policy.actions_prob_distribution.entropy())
        entropy_loss = self.params.entropy_loss_coeff * entropy_loss

        a2c_loss = action_loss + entropy_loss + value_loss
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = regularization_loss + a2c_loss

        # training
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
        self.train = optimizer.minimize(loss, global_step=global_step)

        # summaries for the agent and the training process
        with tf.name_scope('agent_summary'):
            tf.summary.histogram('actions', self.policy.actions)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.policy.act)))

            tf.summary.histogram('selected_actions', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.actions_prob_distribution.entropy()))

            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('entropy_loss', entropy_loss)
            tf.summary.scalar('a2c_loss', a2c_loss)
            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('loss', loss)

            self.summary_writer = tf.summary.FileWriter(summaries_dir())
            self.all_summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

    def _maybe_print(self, step, avg_rewards, fps):
        if step % self.params.print_every == 0:
            logger.info('<====== Step %d ======>', step)
            logger.info('FPS: %.1f', fps)
            logger.info('Avg. 100 episode reward: %.3f', avg_rewards)

    def best_action(self, observation):
        actions, _ = self._policy_step([observation])
        return actions[0]

    def _policy_step(self, observations):
        actions, values = self.session.run(
            [self.policy.act, self.policy.value],
            feed_dict={self.policy.observations: observations},
        )
        return actions, values

    def _estimate_values(self, observations):
        values = self.session.run(
            self.policy.value,
            feed_dict={self.policy.observations: observations},
        )
        return values

    def _train_step(self, step, observations, actions, values, discounted_rewards):
        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        summaries = [self.all_summaries] if with_summaries else []
        result = self.session.run(
            [self.train] + summaries,
            feed_dict={
                self.policy.observations: observations,
                self.selected_actions: actions,
                self.value_estimates: values,
                self.discounted_rewards: discounted_rewards,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[1]
            self.summary_writer.add_summary(summary, global_step=step)

        return step

    @staticmethod
    def _calc_discounted_rewards(gamma, rewards, dones, last_value):
        cumulative = 0 if dones[-1] else last_value
        discounted_rewards = []
        for rollout_step in reversed(range(len(rewards))):
            r, done = rewards[rollout_step], dones[rollout_step]
            cumulative = r + gamma * cumulative * (not done)
            discounted_rewards.append(cumulative)
        return reversed(discounted_rewards)

    def learn(self, multi_env, step_callback=None):
        step = tf.train.global_step(self.session, tf.train.get_global_step())
        training_started = time.time()
        batch_size = self.params.rollout * self.params.num_envs

        observations = multi_env.initial_observations()

        end_of_training = lambda s: s >= self.params.train_for_steps
        while not end_of_training(step):
            batch_obs = [observations]
            batch_actions, batch_values, batch_rewards, batch_dones = [], [], [], []
            for rollout_step in range(self.params.rollout):
                actions, values = self._policy_step(observations)
                batch_actions.append(actions)
                batch_values.append(values)

                observations, rewards, dones = multi_env.step(actions)
                batch_rewards.append(rewards)
                batch_dones.append(dones)

                if rollout_step != self.params.rollout - 1:
                    # we don't need the newest observation in the training batch, already have enough
                    batch_obs.append(observations)

            assert len(batch_obs) == len(batch_rewards)

            batch_rewards = np.asarray(batch_rewards, np.float32).swapaxes(0, 1)
            batch_dones = np.asarray(batch_dones, np.bool).swapaxes(0, 1)
            last_values = self._estimate_values(observations)

            gamma = self.params.gamma
            discounted_rewards = []
            for env_rewards, env_dones, last_value in zip(batch_rewards, batch_dones, last_values):
                discounted_rewards.extend(self._calc_discounted_rewards(gamma, env_rewards, env_dones, last_value))

            batch_obs_shape = (self.params.rollout * multi_env.num_envs, ) + observations[0].shape
            batch_obs = np.asarray(batch_obs, np.float32).swapaxes(0, 1).reshape(batch_obs_shape)
            batch_actions = np.asarray(batch_actions, np.int32).swapaxes(0, 1).flatten()
            batch_values = np.asarray(batch_values, np.float32).swapaxes(0, 1).flatten()

            step = self._train_step(step, batch_obs, batch_actions, batch_values, discounted_rewards)
            self._maybe_save(step)

            avg_rewards = multi_env.calc_avg_rewards(n=100)
            fps = (step * batch_size) / (time.time() - training_started)
            self._maybe_print(step, avg_rewards, fps)
            if step_callback is not None:
                step_callback(locals(), globals())
