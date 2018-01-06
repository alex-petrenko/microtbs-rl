import random


class EpsilonGreedy:
    @staticmethod
    def _linear_decay(x1, y1, x2, y2, x):
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return k * x + b

    def exploration_prob(self, step):
        exploration = self._linear_decay(0.0, 1.0, 1e5, 0.3, step)
        if exploration > 0.3:
            return exploration
        exploration = self._linear_decay(1e5, 0.3, 3e5, 0.1, step)
        if exploration > 0.1:
            return exploration
        return 0.1  # minimum exploration

    def action(self, step, explore, exploit):
        if random.random() < self.exploration_prob(step):
            return explore()
        return exploit()
