"""
Exploration strategy implementation.

"""


import random


class LinearDecay:
    def __init__(self, milestones):
        """
        Linear decay of probability.
        :param milestones: list
        List of tuples (fraction of training, probability)
        E.g. [(0.2, 0.5), (0.6, 0)] means start with 100%, at 20% training linearly decay to 50%, at 60% training
        decay to 0%, stay at 0% until the end.
        """
        self.schedule = sorted(milestones)

    def at(self, fraction):
        if fraction > 1:
            pass

        eps = 1e-10
        assert 0 <= fraction <= 1 + eps
        schedule = [(0.0, 1.0)] + self.schedule + [(1.0 + eps, 0.0)]

        # find where we are in terms of milestones
        milestone = 0
        while schedule[milestone][0] <= fraction:
            milestone += 1

        x = fraction
        x0, y0 = schedule[milestone - 1]
        x1, y1 = schedule[milestone]
        # linear interpolation
        y = y0 * (1 - (x-x0)/(x1-x0)) + y1 * (1 - (x1-x)/(x1-x0))

        probability = y
        return probability


class EpsilonGreedy:
    def __init__(self, schedule):
        self.schedule = schedule

    def exploration_prob(self, fraction_of_training):
        return self.schedule.at(fraction_of_training)

    def action(self, fraction_of_training, explore, exploit):
        if random.random() < self.exploration_prob(fraction_of_training):
            return explore()
        return exploit()
