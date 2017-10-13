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
    world = input_state['world']
    hero = input_state['hero']
    hero_pos = hero['pos']
    size = world.shape[0]
    channels = 2

    state = np.ndarray(shape=(size, size, channels), dtype=np.float32)

    world = world.astype(np.float32)
    world /= np.max(world)  # convert to 0-1 range
    state[:, :, 0] = world  # 1st channel is world tiles (ground, obstacle, gold)

    state[:, :, 1] = 0
    state[hero_pos.i, hero_pos.j, 1] = 1  # 2nd channel codes the position of the player

    return state


class AgentDqn(Agent):
    _saver_dir = './.sessions'
    _summary_dir = './.summary'

    def __init__(self, allowed_actions, game_state):
        super(AgentDqn, self).__init__(allowed_actions)

        self.session = None  # actually created in "initialize" method
        self.memory = ReplayMemory()
        self.exploration_strategy = EpsilonGreedy()

        gamma = 0.95  # future reward discount
        num_actions = len(allowed_actions)
        with_regularization = True

        assert game_state.ndim == 3
        assert game_state.shape[0] == game_state.shape[1]
        input_size = game_state.shape[0]
        input_channels = game_state.shape[2]

        global_step = tf.contrib.framework.get_or_create_global_step()

        self.keep_prob = tf.placeholder(tf.float32)
        if with_regularization:
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
            shape=(None, input_size, input_size, input_channels),
        )

        conv1 = conv(self.state_input, 16, 4)
        conv2 = conv(conv1, 16, 3)
        drop2 = tf.nn.dropout(conv2, self.keep_prob)
        conv3 = conv(drop2, 32, 3)
        flat3 = tf.contrib.layers.flatten(conv3)

        fc1 = fully_connected(flat3, 128)
        fc2 = fully_connected(fc1, 128)

        Q = tf.contrib.layers.fully_connected(fc2, num_actions, activation_fn=None)
        self.action = tf.argmax(Q, axis=1)

        tf.summary.histogram('Q', Q)
        tf.summary.scalar('Q_mean', tf.reduce_mean(Q))
        tf.summary.histogram('action', self.action)
        tf.summary.scalar('action_mean', tf.reduce_mean(self.action))

        logger.info('Total parameters in the model: %d', count_total_parameters())

        self.selected_action = tf.placeholder(tf.int32, shape=[None])
        tf.summary.histogram('selected_action', self.selected_action)
        tf.summary.scalar('selected_action_mean', tf.reduce_mean(self.selected_action))

        action_one_hot = tf.one_hot(self.selected_action, num_actions, 1.0, 0.0)
        self.Q_acted = tf.reduce_sum(Q * action_one_hot, axis=1)

        self.reward = tf.placeholder(tf.float32, shape=[None])
        tf.summary.scalar('reward_mean', tf.reduce_mean(self.reward))

        self.Q_target = tf.placeholder(tf.float32, shape=[None])
        tf.summary.scalar('Q_target_mean', tf.reduce_mean(self.Q_target))

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

            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('discrepancy', tf.reduce_mean(discrepancy))
            tf.summary.scalar('clipped_error', tf.reduce_mean(clipped_error))
            tf.summary.scalar('Q_loss', Q_loss)
            tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        self.train = optimizer.minimize(loss, global_step=global_step)

        with tf.variable_scope('summary'):
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
        # start the session
        self.session = tf.Session()
        try:
            self.saver.restore(
                self.session, tf.train.latest_checkpoint(checkpoint_dir=self._saver_dir),
            )
        except ValueError:
            logger.info('Could not restore, start from scratch')
            self.session.run(tf.global_variables_initializer())

        logger.info('Initialized!')

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
                self.keep_prob: 1,
                self.state_input: state_batch,
            },
        )
        return action_idx

    def act(self, state):
        action_idx = self._best_action([preprocess_state(state)])
        action = self.allowed_actions[action_idx[0]]
        logger.info('Selected action: %r', action)
        return action

    def explore(self, env, curr_state):
        step = tf.train.global_step(self.session, tf.train.get_global_step())
        state = preprocess_state(curr_state)

        action_idx = self.exploration_strategy.action(
            step=step,
            explore=self.random_action,
            exploit=lambda: self._best_action([state])[0],
        )

        new_state, reward = env.step(self.allowed_actions[action_idx])
        self.memory.remember((state, action_idx, preprocess_state(new_state), reward))
        return new_state

    def update(self):
        step = tf.train.global_step(self.session, tf.train.get_global_step())

        if not self.memory.good_enough():
            return  # skip updating until we gain more experience

        state, action_idx, new_state, reward = self.memory.recollect()

        action_idx_new_state = self._best_action(new_state)
        Q_new_state = self.session.run(
            self.Q_acted,
            feed_dict={
                self.keep_prob: 1,
                self.selected_action: action_idx_new_state,
                self.state_input: new_state,
            },
        )

        _, summary = self.session.run(
            [self.train, self.all_summaries],
            feed_dict={
                self.keep_prob: 0.8,
                self.state_input: state,
                self.reward: reward,
                self.selected_action: action_idx,
                self.Q_target: Q_new_state,
                self.exploration_placeholder: self.exploration_strategy.exploration_prob(step),
            },
        )

        self.summary_writer.add_summary(summary, global_step=step)
        self._maybe_save(step)
