import numpy as np


class ReplayBuffer:
    """
    Thank you: https://github.com/BY571/
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_sample(self, states, actions, rewards):
        episode = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "summed_rewards": sum(rewards),
        }
        self.buffer.append(episode)

    def sort(self):
        # sort buffer
        self.buffer = sorted(
            self.buffer, key=lambda i: i["summed_rewards"], reverse=True
        )
        # keep the max buffer size
        self.buffer = self.buffer[: self.max_size]

    def get_random_samples(self, batch_size):
        # WHY SORTING BEFORE RANDOM SAMPLE ?
        self.sort()
        # THIS MIGHT RETURN DUPLICATES
        # BETTER IDEA USE random.sample
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]

        return batch

    def get_n_best(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)
