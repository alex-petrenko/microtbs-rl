import numpy as np
import tensorflow as tf

from utils import *
from dnn_utils import *
from agent import Agent
from exploration import EpsilonGreedy
from replay_memory import ReplayMemory


logger = logging.getLogger(os.path.basename(__file__))


def preprocess_state(input_state):
    """Convert state dictionary to multichannel numpy array."""
    return input_state['visual_state']


class DeepQNetwork:
    def __init__(self, size, channels, num_actions, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)

            if self._with_regularization():
                regularizer = tf.contrib.layers.l2_regularizer(scale=1e-7)
            else:
                regularizer = None

            def fully_connected(x, size):
                return tf.contrib.layers.fully_connected(
                    x,
                    size,
                    weights_regularizer=regularizer,
                    biases_regularizer=regularizer,
                )

            def conv(x, size, kernel_size, stride=1):
                return tf.contrib.layers.conv2d(
                    x,
                    size,
                    kernel_size,
                    stride=stride,
                    weights_regularizer=regularizer,
                    biases_regularizer=regularizer,
                )

            self.state_input = tf.placeholder(
                tf.float32,
                shape=(None, size, size, channels),
            )

            conv1 = conv(self.state_input, 16, 4)
            conv2 = conv(conv1, 16, 3)
            drop2 = tf.nn.dropout(conv2, self.keep_prob)
            conv3 = conv(drop2, 32, 3)
            flat3 = tf.contrib.layers.flatten(conv3)

            fc1 = fully_connected(flat3, 128)
            fc2 = fully_connected(fc1, 128)

            # "dueling" DQN trick
            with tf.variable_scope('dueling'):
                # "value" means how good or bad current state is
                value_fc = fully_connected(fc2, 128)
                value = tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None)

                advantage_fc = fully_connected(fc2, 128)
                advantage = tf.contrib.layers.fully_connected(
                    advantage_fc, num_actions, activation_fn=None,
                )

                # "average" dueling
                self.Q = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))

            self.Q_best = tf.reduce_max(self.Q, axis=1)

            # summaries for this particular DNN
            if self._with_summaries():
                with tf.variable_scope(self.name + '_dqn_summary'):
                    tf.summary.scalar('value', tf.reduce_mean(value))
                    for ac in range(num_actions):
                        tf.summary.histogram('advantage_' + str(ac), advantage[:, ac])
                    tf.summary.scalar('advantage_avg', tf.reduce_mean(advantage))
                    tf.summary.histogram('Q', self.Q)
                    tf.summary.scalar('Q_avg', tf.reduce_mean(self.Q))

            logger.info('Total parameters in the model: %d', count_total_parameters())

    @staticmethod
    def _with_summaries():
        return True

    @staticmethod
    def _with_regularization():
        return True

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class DeepQNetworkTarget(DeepQNetwork):
    """This network is not trained, just occasionnally updated with regular DQN weights."""
    @staticmethod
    def _with_summaries():
        return False

    @staticmethod
    def _with_regularization():
        return False


class AgentDqn(Agent):
    _saver_dir = './.sessions'
    _summary_dir = './.summary'

    def __init__(self, allowed_actions, game_state):
        super(AgentDqn, self).__init__(allowed_actions)

        self.session = None  # actually created in "initialize" method
        self.memory = ReplayMemory()
        self.exploration_strategy = EpsilonGreedy()

        self.target_update_speed = 0.1  # rate to update target DQN towards primary DQN
        gamma = 0.98  # future reward discount

        num_actions = len(allowed_actions)
        assert game_state.ndim == 3
        assert game_state.shape[0] == game_state.shape[1]
        input_size = game_state.shape[0]
        input_channels = game_state.shape[2]

        global_step = tf.contrib.framework.get_or_create_global_step()

        self.primary_dqn = DeepQNetwork(input_size, input_channels, num_actions, 'primary')
        self.target_dqn = DeepQNetworkTarget(
            input_size, input_channels, num_actions, 'target',
        )

        self.update_target_ops = self._generate_target_update_ops()

        self.action = tf.argmax(self.primary_dqn.Q, axis=1)  # predicted action
        self.selected_action = tf.placeholder(tf.int32, shape=[None])  # action actually taken
        action_one_hot = tf.one_hot(self.selected_action, num_actions, 1.0, 0.0)
        self.Q_acted = tf.reduce_sum(self.primary_dqn.Q * action_one_hot, axis=1)

        self.reward = tf.placeholder(tf.float32, shape=[None])

        # Q-value generated by target DQN
        self.Q_target = tf.placeholder(tf.float32, shape=[None])

        with tf.name_scope('loss'):
            # basically, a Bellman equation update
            Q_updated = self.reward + gamma * self.Q_target
            discrepancy = self.Q_acted - Q_updated
            clipped_error = tf.where(
                tf.abs(discrepancy) < 1.0,
                0.5 * tf.square(discrepancy),
                tf.abs(discrepancy) - 0.5,
            )
            Q_loss = tf.reduce_mean(clipped_error)

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.reduce_sum(regularization_losses)
            loss = Q_loss + regularization_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        self.train = optimizer.minimize(loss, global_step=global_step)

        # summaries for the agent and the training process
        with tf.name_scope('agent_summary'):
            primary_trainables = self.primary_dqn.get_trainable_variables()
            target_trainables = self.target_dqn.get_trainable_variables()
            delta = []
            for primary, target in zip(primary_trainables, target_trainables):
                delta.append(tf.reduce_mean(tf.abs(primary - target)))
            tf.summary.scalar('primary_target_delta', tf.add_n(delta) / len(delta))

            tf.summary.histogram('action', self.action)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.action)))

            tf.summary.histogram('sel_action', self.selected_action)
            tf.summary.scalar(
                'sel_action_avg', tf.reduce_mean(tf.to_float(self.selected_action)),
            )

            tf.summary.scalar('reward', tf.reduce_mean(self.reward))
            tf.summary.scalar('Q_target_avg', tf.reduce_mean(self.Q_target))

            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('discrepancy', tf.reduce_mean(discrepancy))
            tf.summary.scalar('clipped_error', tf.reduce_mean(clipped_error))
            tf.summary.scalar('Q_loss', Q_loss)
            tf.summary.scalar('loss', loss)

            total_reward = tf.Variable(0.0)
            avg_reward = tf.Variable(0.0)
            update_total_reward = tf.assign_add(total_reward, tf.reduce_sum(self.reward))
            age = tf.to_float(global_step) + 1.0
            update_avg_reward = tf.assign(avg_reward, total_reward / age)
            tf.summary.scalar('total_reward', total_reward)
            tf.summary.scalar('avg_reward', avg_reward)

            self.exploration_placeholder = tf.placeholder(tf.float32)
            tf.summary.scalar('exploration', self.exploration_placeholder)

            self.summary_writer = tf.summary.FileWriter(self._summary_dir)

            with tf.control_dependencies([update_total_reward, update_avg_reward]):
                self.all_summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

    def initialize(self):
        """Start the session."""
        self.session = tf.Session()
        try:
            self.saver.restore(
                self.session, tf.train.latest_checkpoint(checkpoint_dir=self._saver_dir),
            )
        except ValueError:
            logger.info('Could not restore, start from scratch')
            self.session.run(tf.global_variables_initializer())

        logger.info('Initialized!')

    def _generate_target_update_ops(self):
        primary_trainables = self.primary_dqn.get_trainable_variables()
        target_trainables = self.target_dqn.get_trainable_variables()
        tau = self.target_update_speed

        update_ops = []
        assert len(primary_trainables) == len(target_trainables)
        for primary, target in zip(primary_trainables, target_trainables):
            updated_value = primary * tau + target * (1 - tau)
            update_op = target.assign(updated_value)
            update_ops.append(update_op)
        return update_ops

    def _maybe_update_target(self, step):
        update_every = 20000
        if step % update_every == 0:
            self.session.run(self.update_target_ops)

    @classmethod
    def _saver_path(cls):
        return cls._saver_dir + '/' + cls.__name__

    def _maybe_save(self, step):
        save_every = 5000
        if step % save_every == 0:
            logger.info('Step #%d, saving...', step)
            self.saver.save(self.session, self._saver_path(), global_step=step)

    def _best_action(self, state_batch):
        action_idx = self.session.run(
            self.action,
            feed_dict={
                self.primary_dqn.keep_prob: 1,
                self.primary_dqn.state_input: state_batch,
            },
        )
        return action_idx

    def act(self, state):
        action_idx = self._best_action([preprocess_state(state)])
        action = self.allowed_actions[action_idx[0]]
        logger.info('Selected action: %r', action)
        return action

    def _explore(self, env, curr_state):
        step = tf.train.global_step(self.session, tf.train.get_global_step())
        state = preprocess_state(curr_state)

        action_idx = self.exploration_strategy.action(
            step=step,
            explore=self.random_action,
            exploit=lambda: self._best_action([state])[0],
        )

        new_state, reward = env.step(self.allowed_actions[action_idx])
        return state, action_idx, new_state, reward

    def _update_step(self, state, action_idx, new_state, reward):
        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if step % 1000 == 0:
            logger.info('Training step: %r', step)

        Q_new_state = self.session.run(
            self.target_dqn.Q_best,
            feed_dict={
                self.target_dqn.keep_prob: 1,
                self.target_dqn.state_input: new_state,
            },
        )

        # prevent summaries folder from growing too large
        train_ops = [self.train]
        with_summaries = (step % 10 == 0)
        if with_summaries:
            train_ops.append(self.all_summaries)

        result = self.session.run(
            train_ops,
            feed_dict={
                self.primary_dqn.keep_prob: 0.8,
                self.primary_dqn.state_input: state,
                self.reward: reward,
                self.selected_action: action_idx,
                self.Q_target: Q_new_state,
                self.exploration_placeholder: self.exploration_strategy.exploration_prob(step),
            },
        )

        if with_summaries:
            summary = result[1]
            self.summary_writer.add_summary(summary, global_step=step)

        self._maybe_update_target(step)
        self._maybe_save(step)

    def update(self, env, curr_state):
        state, action_idx, new_state, reward = self._explore(env, curr_state)
        self._update_step([state], [action_idx], [preprocess_state(new_state)], [reward])
        return new_state

    def explore_and_remember(self, env, curr_state):
        state, action_idx, new_state, reward = self._explore(env, curr_state)
        self.memory.remember((state, action_idx, preprocess_state(new_state), reward))
        return new_state

    def update_with_replay_memory(self):
        if not self.memory.good_enough():
            return  # skip updating until we gain more experience
        state, action_idx, new_state, reward = self.memory.recollect()
        self._update_step(state, action_idx, new_state, reward)
