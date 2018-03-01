import threading
import numpy as np

from queue import Queue

from utils.dnn_utils import *
from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


class _MultiEnvWorker:
    """
    Helper class for the MultiEnv.
    Currently implemented with threads, and it's slow because of GIL.
    It would be much better to implement this with multiprocessing.
    """

    def __init__(self, idx, make_env_func):
        self.idx = idx

        self.env = make_env_func()
        self.env.seed(idx)

        self.observation = self.env.reset()

        self.action_queue = Queue()
        self.result_queue = Queue()

        self.thread = threading.Thread(target=self.start)
        self.thread.start()

    def start(self):
        while True:
            action = self.action_queue.get()
            if action is None:  # stop signal
                logger.info('Stop worker %d...', self.idx)
                break

            observation, reward, done, _ = self.env.step(action)
            if done:
                observation = self.env.reset()

            self.result_queue.put((observation, reward, done))
            self.action_queue.task_done()


class MultiEnv:
    """Run multiple gym-compatible environments in parallel, keeping more or less the same interface."""

    def __init__(self, num_envs, make_env_func):
        self.num_envs = num_envs
        self.workers = [_MultiEnvWorker(i, make_env_func) for i in range(num_envs)]

        self.action_space = self.workers[0].env.action_space
        self.observation_space = self.workers[0].env.observation_space

        self.curr_episode_reward = [0] * num_envs
        self.episode_rewards = [[]] * num_envs

    def initial_observations(self):
        return [worker.observation for worker in self.workers]

    def step(self, actions):
        """Obviously, returns vectors of obs, rewards, dones instead of usual single values."""
        assert len(actions) == len(self.workers)
        for worker, action in zip(self.workers, actions):
            worker.action_queue.put(action)

        results = []
        for worker in self.workers:
            worker.action_queue.join()
            results.append(worker.result_queue.get())

        observations, rewards, dones = zip(*results)

        for i in range(self.num_envs):
            self.curr_episode_reward[i] += rewards[i]
            if dones[i]:
                self.episode_rewards[i].append(self.curr_episode_reward[i])
                self.curr_episode_reward[i] = 0

        return observations, rewards, dones

    def close(self):
        logger.info('Stopping multi env...')
        for worker in self.workers:
            worker.action_queue.put(None)  # terminate
            worker.thread.join()

    def calc_avg_rewards(self, n):
        avg_reward = 0
        for i in range(self.num_envs):
            last_episodes_rewards = self.episode_rewards[i][-n:]
            avg_reward += np.mean(last_episodes_rewards)
        return avg_reward / float(self.num_envs)
