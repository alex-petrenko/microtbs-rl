import random


class EpsilonGreedy:
    def __init__(self, min_exploration=0.2, decay_steps=200000):
        self.min_exploration = min_exploration
        self.decay_steps = decay_steps

    def exploration_prob(self, step):
        decay = step * (1.0 - self.min_exploration) / self.decay_steps
        return max(1.0 - decay, self.min_exploration)

    def action(self, step, explore, exploit):
        if random.random() < self.exploration_prob(step):
            return explore()
        return exploit()
