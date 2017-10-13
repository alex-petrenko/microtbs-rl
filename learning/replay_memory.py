import numpy as np

from utils import *


logger = logging.getLogger(os.path.basename(__file__))


class ReplayMemory:
    def __init__(self):
        self.batch_size = 256
        self.store_max_batches = 1000
        self.storage = []

    def good_enough(self):
        min_memory = self.batch_size * 50
        if len(self.storage) < min_memory:
            if len(self.storage) % 10 == 0:
                logger.info('Gaining experience, %r/%r', len(self.storage), min_memory)
            return False
        return True

    def remember(self, experience):
        self.storage.append(experience)
        self.forget()

    def forget(self):
        while len(self.storage) > self.batch_size * self.store_max_batches:
            self.storage.pop(0)

    def recollect(self):
        indices = np.random.choice(len(self.storage), self.batch_size, replace=False)
        batch = [self.storage[i] for i in indices]
        batch = tuple(zip(*batch))
        return batch
