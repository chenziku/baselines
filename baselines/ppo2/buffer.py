import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

rgb_weights = [0.2989, 0.5870, 0.1140]

class Buffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, batch):
        if self._next_idx >= len(self._storage) + len(batch):
            self._storage += batch
        else:
            self._storage[self._next_idx:self._next_idx+len(batch)] = batch
        self._next_idx = (self._next_idx + len(batch)) % self._maxsize

    # def _encode_sample(self, idxes):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, action, reward, obs_tp1, done = data
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def compute_stats(self):
        storage_np = np.array(self._storage).reshape((-1, 64*64))
        mu = storage_np.mean(axis=0)
        sd = storage_np.std(axis=0)
        return mu, sd

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
