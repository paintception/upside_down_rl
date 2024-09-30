from dataclasses import dataclass
import gymnasium as gym
import numpy as np

from udrl.catch import CatchAdaptor
from udrl.policies import ABCPolicy
from udrl.buffer import ReplayBuffer


@dataclass
class AgentHyper:
    """Hyperparameters for an agent interacting with an environment.

    Parameters
    ----------
    env_name : str
        Name of the environment the agent interacts with.
    warm_up : int, optional
        Number of initial steps before training begins (default: 50).
    memory_size : int, optional
        Maximum size of the agent's experience replay memory (default: 700).
    last_few : int, optional
        Number of recent experiences to prioritize in training (default: 75).
    batch_size : int, optional
        Number of experiences sampled from memory for each training update
        (default: 32).
    horizon_scale : float, optional
        Scaling factor for the horizon length in reinforcement learning
        (default: 0.02).
    return_scale : float, optional
        Scaling factor for rewards or returns in reinforcement learning
        (default: 0.02).
    """

    env_name: str
    warm_up: int = 50
    memory_size: int = 700
    last_few: int = 75
    batch_size: int = 32

    horizon_scale: float = 0.02
    return_scale: float = 0.02


class UpsideDownAgent:
    """An agent that interacts with an environment using an
    Upside-Down Reinforcement Learning approach.

    Parameters
    ----------
    conf : AgentHyper
        Hyperparameters for the agent.
    policy : ABCPolicy
        A policy object used by the agent to select actions.

    Attributes
    ----------
    environment : gym.Env
        The Gym environment the agent interacts with.
    state_size : int
        The size of the state space in the environment.
    memory : ReplayBuffer
        The replay buffer used to store experiences for training.
    policy : ABCPolicy
        The policy object used by the agent to select actions.

    Methods
    -------
    collect_episode(desired_return=1, desired_horizon=1, random=False,
                    store_episode=True, test=False)
        Collects an episode of experience from the environment.
    sample_exploratory_commands()
        Samples exploratory commands based on past experiences.
    train()
        Trains the agent's policy using experiences from the replay buffer.
    """

    def __init__(self, conf: AgentHyper, policy: ABCPolicy):
        self.conf = conf
        self.environment = (
            CatchAdaptor(dense=True)
            if conf.env_name == "catch"
            else gym.make(conf.env_name)
        )
        self.state_size = self.environment.observation_space.shape[0]
        self.memory = ReplayBuffer(conf.memory_size)
        self.policy = policy
        for x in range(conf.warm_up):
            self.collect_episode(random=True)

    def collect_episode(
        self,
        desired_return: int = 1,
        desired_horizon: int = 1,
        random: bool = False,
        store_episode: bool = True,
        test: bool = False,
    ):
        state, _ = self.environment.reset()
        epochs = []
        horizons = []
        returns = []
        cum_rew = 0
        tru, ter = False, False

        while not (tru or ter):
            state = np.expand_dims(state, axis=0)
            command = np.array(
                [
                    desired_return * self.conf.return_scale,
                    desired_horizon * self.conf.horizon_scale,
                ]
            )
            command = np.expand_dims(command, axis=0)
            action = (
                self.environment.action_space.sample()
                if random
                else self.policy(state, command, test)
            )
            next_state, reward, tru, ter, _ = self.environment.step(action)

            epochs.append([state, action, reward])
            cum_rew += reward
            horizons.append(desired_horizon)
            returns.append(desired_return)

            state = next_state
            # Line 8 Algorithm 2
            desired_return -= reward
            # Line 9 Algorithm 2
            desired_horizon = max(desired_horizon - 1, 1)
        if store_episode:
            self.memory.add_sample(*list(zip(*epochs)))
        return cum_rew, returns, horizons

    def sample_exploratory_commands(self):
        best_ep = self.memory.get_n_best(self.conf.last_few)
        expl_desired_horizon = np.mean([len(i["states"]) for i in best_ep])

        returns = [i["summed_rewards"] for i in best_ep]
        expl_desired_returns = np.random.uniform(
            np.mean(returns), np.mean(returns) + np.std(returns)
        )

        return [expl_desired_returns, expl_desired_horizon]

    def train(self):
        batch_size = self.conf.batch_size
        if self.conf.batch_size <= 0:
            batch_size = len(self.memory.buffer)

        random_episodes = self.memory.get_random_samples(batch_size)

        training_states = np.zeros((batch_size, self.state_size))
        training_commands = np.zeros((batch_size, 2))

        actions = []

        for idx, episode in enumerate(random_episodes):
            T = len(episode["states"])
            t1 = np.random.randint(0, T - 1)
            # t2 = np.random.randint(t1 + 1, T)
            t2 = T

            state = episode["states"][t1]
            desired_return = sum(episode["rewards"][t1:t2])
            desired_horizon = t2 - t1

            action = episode["actions"][t1]

            training_states[idx] = state[0]
            training_commands[idx] = np.array(
                [
                    desired_return * self.conf.return_scale,
                    desired_horizon * self.conf.horizon_scale,
                ]
            )
            actions.append(action)

        return self.policy.train(training_states, training_commands, actions)
