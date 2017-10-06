import random


class Agent:
    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions

    def act(self, state):
        raise NotImplementedError(self)


class AgentRandom(Agent):
    def act(self, state):
        idx = random.randrange(len(self.allowed_actions))
        return self.allowed_actions[idx]
