import shutil
import unittest

from algorithms.baselines import enjoy_baseline_random
from algorithms.baselines.openai_baselines import enjoy_baseline_dqn
from algorithms.baselines.openai_baselines import train_baseline_dqn

from utils.common_utils import get_test_logger, experiment_dir


logger = get_test_logger()

TEST_ENV = 'MicroTbs-CollectWithTerrain-v2'


class RandomBaselineTest(unittest.TestCase):
    def test_random_baseline(self):
        self.assertEqual(enjoy_baseline_random.enjoy(TEST_ENV, max_num_episodes=1, fps=500), 0)


class OpenaiDqnBaselineTest(unittest.TestCase):
    def test_train(self):
        experiment_name = 'openai_dqn_baseline_test'
        self.assertEqual(train_baseline_dqn.train(experiment_name, TEST_ENV, 10), 0)
        shutil.rmtree(experiment_dir(experiment_name))
