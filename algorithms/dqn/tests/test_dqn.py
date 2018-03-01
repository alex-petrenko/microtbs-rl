"""
DQN agent tests.

"""

import os
import gym
import shutil
import logging
import unittest
import numpy as np

import envs

from algorithms import dqn

from utils.common_utils import get_test_logger, experiment_dir


logger = get_test_logger()

TEST_ENV = envs.COLLECT_WITH_TERRAIN_LATEST


class DqnTest(unittest.TestCase):
    def test_train_and_run(self):
        experiment_name = 'dqn_test'
        dqn_params = dqn.AgentDqn.Params(experiment_name)
        dqn_params.train_for_steps = 10
        dqn_params.save_every = dqn_params.train_for_steps - 1
        self.assertEqual(dqn.train_dqn.train(dqn_params, TEST_ENV), 0)
        self.assertEqual(dqn.enjoy_dqn.enjoy(experiment_name, TEST_ENV, max_num_episodes=1, fps=500), 0)
        shutil.rmtree(experiment_dir(experiment_name))
