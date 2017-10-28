import random

from utils import *


logger = logging.getLogger(os.path.basename(__file__))


class ReplayMemory:
    def __init__(self):
        self.store_max_episodes = 10000
        self.episodes = []

    def good_enough(self):
        min_memory = 50
        if len(self.episodes) < min_memory:
            return False
        return True

    def remember(self, experience):
        self.episodes.append(experience)
        self.forget()

    def forget(self):
        while len(self.episodes) > self.store_max_episodes:
            idx_to_delete = random.randrange(0, len(self.episodes))
            self.episodes.pop(idx_to_delete)

    def recollect(self, batch_size, temporal_rollout=1):
        batch = []
        while len(batch) < batch_size * temporal_rollout:
            episode = random.sample(self.episodes, 1)[0]
            if len(episode) < temporal_rollout:
                logger.info('Episode is too short! %d/%d', len(episode), temporal_rollout)
                continue

            point_in_episode = random.randint(0, len(episode) - temporal_rollout)
            end_point = point_in_episode + temporal_rollout
            rollout = episode[point_in_episode:end_point]
            batch.extend(rollout)

        batch = tuple(zip(*batch))
        return batch
