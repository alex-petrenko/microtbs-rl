import os
import logging

from unittest import TestCase

from algorithms import a2c

import envs
from utils.exploration import LinearDecay
from utils.common_utils import get_test_logger
from utils.record_policy_execution import record


logger = get_test_logger()


class ExplorationTests(TestCase):
    def test_linear_decay(self):
        schedule = LinearDecay(milestones=[])
        self.assertAlmostEqual(schedule.at(0), 1)
        self.assertAlmostEqual(schedule.at(1), 0)
        self.assertAlmostEqual(schedule.at(0.3), 0.7)
        self.assertAlmostEqual(schedule.at(0.5), 0.5)
        self.assertAlmostEqual(schedule.at(0.7), 0.3)

        schedule = LinearDecay(milestones=[(0.5, 0)])
        self.assertAlmostEqual(schedule.at(0), 1)
        self.assertAlmostEqual(schedule.at(1), 0)
        self.assertAlmostEqual(schedule.at(0.25), 0.5)
        self.assertAlmostEqual(schedule.at(0.5), 0)
        self.assertAlmostEqual(schedule.at(0.7), 0)

        schedule = LinearDecay(milestones=[(0.4, 0.2), (0.8, 0.1)])
        self.assertAlmostEqual(schedule.at(0), 1)
        self.assertAlmostEqual(schedule.at(1), 0)
        self.assertAlmostEqual(schedule.at(0.1), 0.8)
        self.assertAlmostEqual(schedule.at(0.4), 0.2)
        self.assertAlmostEqual(schedule.at(0.6), 0.15)
        self.assertAlmostEqual(schedule.at(0.9), 0.05)


class RenderGifTests(TestCase):
    def test_render_gif(self):
        experiment_name = 'gif_test'
        test_env = envs.COLLECT_WITH_TERRAIN_LATEST
        a2c_params = a2c.AgentA2C.Params(experiment_name)
        a2c_params.train_for_steps = 10
        a2c_params.save_every = a2c_params.train_for_steps - 1
        self.assertEqual(a2c.train_a2c.train(a2c_params, test_env), 0)
        self.assertEqual(record(experiment_name, test_env, num_episodes=1), 0)
        shutil.rmtree(experiment_dir(experiment_name))
