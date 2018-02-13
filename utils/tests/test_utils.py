import os
import logging

from unittest import TestCase

from utils.exploration import LinearDecay

from utils.common_utils import get_test_logger


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
