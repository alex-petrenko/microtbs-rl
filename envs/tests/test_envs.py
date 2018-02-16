import os
import gym
import unittest
import numpy as np

import envs


class MicroTbsTests(unittest.TestCase):
    def test_reproducibility_rng_seed(self):
        def generate_env():
            env = gym.make('MicroTbs-CollectSimple-v1')
            env.seed(0)
            return env

        env1, env2 = generate_env(), generate_env()
        for i in range(100):
            # make sure generated worlds are the same
            self.assertTrue(np.array_equal(env1.reset(), env2.reset()))
