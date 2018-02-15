import json
import numpy as np
import tensorflow as tf

from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Agent:
    class Params:
        def __init__(self, experiment_name):
            self.experiment_name = experiment_name
            self._params_serialized = False

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

    def __init__(self, params):
        self.params = params

    def initialize(self):
        pass

    def finalize(self):
        pass

    def analyze_observation(self, observation):
        """Default implementation, may be or may not be overridden."""
        return None

    def best_action(self, observation):
        """Must be overridden in derived classes."""
        raise NotImplementedError('Subclasses should implement {}'.format(self.best_action.__name__))


class AgentRandom(Agent):
    def __init__(self, params, env):
        super(AgentRandom, self).__init__(params)
        self.action_space = env.action_space

    def best_action(self, _):
        return self.action_space.sample()


# noinspection PyAbstractClass
class AgentLearner(Agent):
    def __init__(self, params):
        super(AgentLearner, self).__init__(params)
        self.session = None  # actually created in "initialize" method
        self.saver = None
        tf.reset_default_graph()

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

    def finalize(self):
        self.session.close()

    def _maybe_save(self, step):
        next_step = step + 1
        self.params.ensure_serialized()
        if next_step % self.params.save_every == 0:
            logger.info('Step #%d, saving...', step)
            saver_path = model_dir(self.params.experiment_name) + '/' + self.__class__.__name__
            self.saver.save(self.session, saver_path, global_step=step)
