import json
import numpy as np

from utils.dnn_utils import *
from utils.common_utils import *

from utils.replay_memory import ReplayMemory
from utils.exploration import EpsilonGreedy, LinearDecay


logger = logging.getLogger(os.path.basename(__file__))


class DeepQNetwork:
    def __init__(self, input_shape, num_actions, name, is_target=False):
        self.name = name

        with_regularization = with_summaries = not is_target

        with tf.variable_scope(self.name):
            self.regularizer = None
            if with_regularization:
                self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

            # process observations
            input_shape = [None] + input_shape  # add batch dimension
            self.observations = tf.placeholder(tf.float32, shape=input_shape)

            # convolutions
            # conv_filters = self.convnet_simple([(32, 3, 1), (64, 3, 1), (64, 3, 1)])
            conv_filters = self.inception()
            conv_out = tf.contrib.layers.flatten(conv_filters)

            # "dueling" DQN trick
            with tf.variable_scope('dueling'):
                # value is supposed to mean how good or bad the current state is
                value_fc1 = dense(conv_out, 256, self.regularizer)
                value_fc2 = dense(value_fc1, 256, self.regularizer)
                self.value = tf.contrib.layers.fully_connected(value_fc2, 1, activation_fn=None)

                # advantage is a score assigned to each action
                advantage_fc1 = dense(conv_out, 256, self.regularizer)
                advantage_fc2 = dense(advantage_fc1, 256, self.regularizer)
                self.advantage = tf.contrib.layers.fully_connected(advantage_fc2, num_actions, activation_fn=None)

                # average dueling
                self.Q = self.value + (self.advantage - tf.reduce_mean(self.advantage, axis=1, keep_dims=True))

            self.Q_best = tf.reduce_max(self.Q, axis=1)
            self.best_action = tf.argmax(self.Q, axis=1)

            # summaries for this particular net
            if with_summaries:
                with tf.variable_scope('conv1', reuse=True):
                    weights = tf.get_variable('weights')
                    tf.summary.image('conv1/kernels', put_kernels_on_grid(weights), max_outputs=1)
                tf.summary.scalar('value', tf.reduce_mean(self.value))
                for ac in range(num_actions):
                    tf.summary.histogram('advantage_' + str(ac), self.advantage[:, ac])
                tf.summary.scalar('advantage_avg', tf.reduce_mean(self.advantage))
                tf.summary.histogram('Q', self.Q)
                tf.summary.scalar('Q_avg', tf.reduce_mean(self.Q))

            logger.info('Total parameters in the model: %d', count_total_parameters())

    def conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self.regularizer, scope=scope)

    def convnet_simple(self, convs):
        layer = self.observations
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self.conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer

    def inception(self):
        conv_input = self.convnet_simple([(128, 3, 1)])

        with tf.variable_scope('branch1x1'):
            branch1x1 = self.conv(conv_input, 64, 1, 1)
        with tf.variable_scope('branch5x5'):
            branch5x5 = self.conv(conv_input, 48, 1, 1)
            branch5x5 = self.conv(branch5x5, 64, 5, 1)
        with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = self.conv(conv_input, 64, 1, 1)
            branch3x3dbl = self.conv(branch3x3dbl, 96, 3, 1)
            branch3x3dbl = self.conv(branch3x3dbl, 96, 3, 1)
        with tf.variable_scope('branch_pool'):
            branch_pool = avg_pool(conv_input, 3, 1)
            branch_pool = self.conv(branch_pool, 32, 1, 1)

        return tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class AgentDqnSimple:
    class Params:
        def __init__(self, experiment_name):
            self.experiment_name = experiment_name

            self.target_update_speed = 0.05  # rate to update target DQN towards primary DQN
            self.update_target_every = 100  # apply target update ops every N training steps
            self.gamma = 0.9  # future reward discount in Bellman equation
            self.learning_rate = 1e-3

            self.save_every = 5000
            self.summaries_every = 100
            self.print_every = 500

            self.train_for_steps = 50000
            self.episodes_in_buffer_before_training = 100
            self.batch_size = 32

            self.exploration_schedule = [(0.8, 0)]  # in 80% of steps decay from 100% to 0% exploration

        def _params_file(self):
            return join(experiment_dir(self.experiment_name), 'params.json')

        def serialize(self):
            with open(self._params_file(), 'w') as json_file:
                json.dump(self.__dict__, json_file, indent=2)

        def load(self):
            with open(self._params_file()) as json_file:
                self.__dict__ = json.load(json_file)
                return self

    def __init__(self, env, params):
        self.params = params

        self.session = None  # actually created in "initialize" method
        self.memory = ReplayMemory()
        self.exploration_strategy = EpsilonGreedy(LinearDecay(milestones=self.params.exploration_schedule))
        self.episode_buffer = []

        global_step = tf.train.get_or_create_global_step()

        # create neural networks that we shall be training
        input_shape = list(env.observation_space.shape)
        num_actions = env.action_space.n
        self.primary_dqn = DeepQNetwork(input_shape, num_actions, 'primary')
        self.target_dqn = DeepQNetwork(input_shape, num_actions, 'target', is_target=True)

        # operations for updating the target network
        self.update_target_ops = self._generate_target_update_ops()

        # operations and inputs needed for Q-learning algorithm
        self.selected_actions = tf.placeholder(tf.int32, shape=[None])  # action actually taken
        action_one_hot = tf.one_hot(self.selected_actions, num_actions, 1.0, 0.0)
        self.Q_predicted = tf.reduce_sum(self.primary_dqn.Q * action_one_hot, axis=1)  # score of the selected action

        # reward returned by the environment
        self.rewards = tf.placeholder(tf.float32, shape=[None])

        # Q-value for the best action in a new env state, predicted by "target" DQN
        self.Q_best = tf.placeholder(tf.float32, shape=[None])

        # On the episode's last step we should not add the predicted best Q-value. Because there's no more reward in
        # the future. Use this mask to disable the Q scores we don't need.
        self.dones = tf.placeholder(tf.float32, shape=[None])

        # basically, a Bellman equation update
        Q_best_masked = (1.0 - self.dones) * self.Q_best
        Q_updated = self.rewards + self.params.gamma * Q_best_masked

        # loss function
        discrepancy = self.Q_predicted - Q_updated
        clipped_error = tf.where(
            tf.abs(discrepancy) < 1.0,
            0.5 * tf.square(discrepancy),
            tf.abs(discrepancy) - 0.5,
        )
        Q_loss = tf.reduce_mean(clipped_error)  # error in predicting the correct Q-value
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.reduce_sum(regularization_losses)
        loss = Q_loss + regularization_loss

        # training
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
        self.train = optimizer.minimize(loss, global_step=global_step)

        # summaries for the agent and the training process
        with tf.name_scope('agent_summary'):
            primary_trainables = self.primary_dqn.get_trainable_variables()
            target_trainables = self.target_dqn.get_trainable_variables()
            delta = []
            for primary, target in zip(primary_trainables, target_trainables):
                delta.append(tf.reduce_mean(tf.abs(primary - target)))
            tf.summary.scalar('primary_target_delta', tf.add_n(delta) / len(delta))

            tf.summary.histogram('action', self.primary_dqn.best_action)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.primary_dqn.best_action)))

            tf.summary.histogram('selected_action', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('reward', tf.reduce_mean(self.rewards))
            tf.summary.scalar('Q_target_avg', tf.reduce_mean(self.Q_best))

            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('discrepancy', tf.reduce_mean(discrepancy))
            tf.summary.scalar('clipped_error', tf.reduce_mean(clipped_error))
            tf.summary.scalar('Q_loss', Q_loss)
            tf.summary.scalar('loss', loss)

            total_reward = tf.Variable(0.0)
            avg_reward = tf.Variable(0.0)
            update_total_reward = tf.assign_add(total_reward, tf.reduce_sum(self.rewards))
            age = tf.to_float(global_step) + 1.0
            # noinspection PyTypeChecker
            update_avg_reward = tf.assign(avg_reward, total_reward / age)
            tf.summary.scalar('total_reward', total_reward)
            tf.summary.scalar('avg_reward', avg_reward)

            self.exploration_placeholder = tf.placeholder(tf.float32)
            tf.summary.scalar('exploration', self.exploration_placeholder)

            self.summary_writer = tf.summary.FileWriter(summaries_dir())

            with tf.control_dependencies([update_total_reward, update_avg_reward]):
                self.all_summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

    def initialize(self):
        """Start the session."""
        self.session = tf.Session()
        checkpoint_dir = model_dir(self.params.experiment_name)
        try:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            logger.info('Didn\'t find a valid restore point, start from scratch')
            self.session.run(tf.global_variables_initializer())
        logger.info('Initialized!')

    def _generate_target_update_ops(self):
        primary_trainables = self.primary_dqn.get_trainable_variables()
        target_trainables = self.target_dqn.get_trainable_variables()
        tau = self.params.target_update_speed

        update_ops = []
        assert len(primary_trainables) == len(target_trainables)
        for primary, target in zip(primary_trainables, target_trainables):
            updated_value = primary * tau + target * (1 - tau)
            update_op = target.assign(updated_value)
            update_ops.append(update_op)
        return update_ops

    def _maybe_update_target(self, step):
        if step % self.params.update_target_every == 0:
            self.session.run(self.update_target_ops)

    def _maybe_save(self, step):
        next_step = step + 1
        if next_step % self.params.save_every == 0:
            logger.info('Step #%d, saving...', step)
            saver_path = model_dir(self.params.experiment_name) + '/' + self.__class__.__name__
            self.saver.save(self.session, saver_path, global_step=step)
            self.params.serialize()

    def _maybe_print(self, step, rewards):
        if step % self.params.print_every == 0:
            exploration = self.exploration_strategy.exploration_prob(self._train_fraction(step))
            logger.info('<====== Step %d ======>', step)
            logger.info('Exploration %%: %.1f', exploration * 100)
            logger.info('Avg. 100 episode reward: %.3f', np.mean(rewards[-100:]))

    def _on_new_episode(self):
        # add the last episode to replay memory
        if self.episode_buffer:
            self.memory.remember(self.episode_buffer)

        # empty the episode buffer
        self.episode_buffer = []

    def _train_fraction(self, step):
        return step / self.params.train_for_steps

    def analyze_observation(self, observation):
        Q, value, advantage = self.session.run(
            [self.primary_dqn.Q, self.primary_dqn.value, self.primary_dqn.advantage],
            feed_dict={self.primary_dqn.observations: [observation]},
        )
        return {'q': Q[0], 'value': value[0], 'adv': advantage[0]}

    def best_action(self, observation):
        actions = self.session.run(
            self.primary_dqn.best_action,
            feed_dict={self.primary_dqn.observations: [observation]},
        )
        return actions[0]

    def _explore(self, env, observation, step):
        action = self.exploration_strategy.action(
            fraction_of_training=self._train_fraction(step),
            explore=env.action_space.sample,
            exploit=lambda: self.best_action(observation),
        )

        new_observation, reward, done, _ = env.step(action)
        experience = (observation, action, new_observation, reward, done)
        self.episode_buffer.append(experience)
        return new_observation, reward, done

    def _train_step(self, step):
        batch_size = self.params.batch_size
        observations, actions, new_observations, rewards, dones = self.memory.recollect(batch_size=batch_size)

        # select the best Q-value in the new env state
        Q_new_state = self.session.run(
            self.target_dqn.Q_best, feed_dict={self.target_dqn.observations: new_observations},
        )

        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        summaries = [self.all_summaries] if with_summaries else []
        result = self.session.run(
            [self.train] + summaries,
            feed_dict={
                self.primary_dqn.observations: observations,
                self.selected_actions: actions,
                self.rewards: rewards,
                self.dones: dones,
                self.Q_best: Q_new_state,
                self.exploration_placeholder: self.exploration_strategy.exploration_prob(self._train_fraction(step)),
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[1]
            self.summary_writer.add_summary(summary, global_step=step)

        self._maybe_update_target(step)
        return step

    def learn(self, env, step_callback=None):
        episode_rewards = []
        step = tf.train.global_step(self.session, tf.train.get_global_step())

        end_of_training = lambda s: s >= self.params.train_for_steps
        while not end_of_training(step):
            observation = env.reset()
            self._on_new_episode()
            episode_rewards.append(0)
            episode_done = False

            while not episode_done and not end_of_training(step):
                observation, reward, episode_done = self._explore(env, observation, step)
                episode_rewards[-1] += reward

                if self.memory.size() >= self.params.episodes_in_buffer_before_training:
                    step = self._train_step(step)
                    self._maybe_save(step)
                    self._maybe_print(step, episode_rewards)
                    if step_callback is not None:
                        step_callback(locals(), globals())
