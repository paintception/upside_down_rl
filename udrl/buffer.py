import random


class ReplayBuffer:
    """A replay buffer for storing and sampling experiences.
    Thank you: https://github.com/BY571/

    Parameters
    ----------
    max_size : int
        The maximum number of experiences the buffer can store.

    Attributes
    ----------
    max_size : int
        The maximum number of experiences the buffer can store.
    buffer : list
        The list storing the experiences.

    Methods
    -------
    add_sample(states, actions, rewards)
        Adds an episode of experience to the buffer and sorts the buffer
        by summed rewards in descending order.

    sort()
        Sorts the buffer by summed rewards in descending order and keeps only
        the top `max_size` experiences.

    get_random_samples(batch_size)
        Returns a random sample of `batch_size` experiences from the buffer.

    get_n_best(n)
        Returns the `n` experiences with the highest summed rewards.

    __len__()
        Returns the current number of experiences in the buffer.
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
        self.sort()

    def sort(self):
        # sort buffer
        self.buffer = sorted(
            self.buffer, key=lambda i: i["summed_rewards"], reverse=True
        )
        # keep the max buffer size
        self.buffer = self.buffer[: self.max_size]

    def get_random_samples(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_n_best(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)
