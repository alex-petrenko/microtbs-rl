import shutil
import unittest

from microtbs_rl import envs

from microtbs_rl.utils.common_utils import get_test_logger, experiment_dir

from microtbs_rl.algorithms.baselines import enjoy_baseline_random
from microtbs_rl.algorithms.baselines.openai_baselines import train_baseline_dqn


logger = get_test_logger()

TEST_ENV = envs.COLLECT_WITH_TERRAIN_LATEST


class RandomBaselineTest(unittest.TestCase):
    def test_random_baseline(self):
        self.assertEqual(enjoy_baseline_random.enjoy(TEST_ENV, max_num_episodes=1, fps=500), 0)


class OpenaiDqnBaselineTest(unittest.TestCase):
    def test_train(self):
        experiment_name = 'openai_dqn_baseline_test'
        self.assertEqual(train_baseline_dqn.train(experiment_name, TEST_ENV, 10), 0)
        shutil.rmtree(experiment_dir(experiment_name))
