"""
Some tests for the MicroTbs environment.

"""

import gym
import unittest
import numpy as np

from microtbs_rl import envs

TEST_ENV = envs.COLLECT_SIMPLE_LATEST


class MicroTbsTests(unittest.TestCase):
    def test_reproducibility_rng_seed(self):
        def generate_env():
            env = gym.make(TEST_ENV)
            env.seed(0)
            return env

        env1, env2 = generate_env(), generate_env()
        for i in range(100):
            # make sure generated worlds are the same
            self.assertTrue(np.array_equal(env1.reset(), env2.reset()))

    def test_render_rgb_array(self):
        env = gym.make(TEST_ENV)
        env.reset()
        array = env.render(mode='rgb_array')
        self.assertIsNotNone(array)
