import random


class Agent:
    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions

    def act(self, state):
        raise NotImplementedError(self)

    def _random_action_idx(self, *_):
        return random.randrange(len(self.allowed_actions))

    def random_action(self, *_):
        return self.allowed_actions[self._random_action_idx()]


class AgentRandom(Agent):
    def act(self, state):
        return self.random_action(state)
