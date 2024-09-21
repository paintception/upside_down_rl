from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import keras

from udrl.policies import ABCPolicy
from udrl.buffer import ReplayBuffer


@dataclass
class AgentHyper:
    env: str
    warm_up: int = 50
    memory_size: int = 700
    last_few: int = 75
    batch_size: int = 32

    horizon_scale: float = 0.02
    return_scale: float = 0.02


class UpsideDownAgent:
    def __init__(self, conf: AgentHyper, policy: ABCPolicy):
        self.conf = conf
        self.environment = gym.make(conf.env)
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
        cum_rew = 0
        tru, ter = False, False

        while not (tru or ter):
            state = np.expand_dims(state, axis=0)

            # interesting continuous scaling -> goes to 0 pretty fast
            # wanted ?
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

            state = next_state
            desired_return -= reward  # Line 8 Algorithm 2
            desired_horizon -= 1  # Line 9 Algorithm 2
            desired_horizon = np.maximum(desired_horizon, 1)
        if store_episode:
            self.memory.add_sample(*list(zip(*epochs)))
        return cum_rew

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
            t2 = np.random.randint(t1 + 1, T)

            state = episode["states"][t1]
            desired_return = sum(episode["rewards"][t1:t2])
            desired_horizon = t2 - t1

            action = episode["actions"][t1]

            training_states[idx] = state[0]
            training_commands[idx] = np.asarray(
                [
                    desired_return * self.conf.return_scale,
                    desired_horizon * self.conf.horizon_scale,
                ]
            )
            actions.append(action)

        actions = keras.utils.to_categorical(actions)
        self.policy.train(training_states, training_commands, actions)


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         self.testing_state += 1

#         feature_importances = {}

#         for t in self.behaviour_function.estimators_:
#             branch = t.decision_path(input_state).todense()
#             branch = np.array(branch, dtype=bool)
#             imp = t.tree_.impurity[branch[0]]
#             for f, i in zip(t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]):
#                 feature_importances.setdefault(f, []).append(i)

#         print(len(feature_importances))
#         summed_importances = [
#             sum(feature_importances[0]),
#             sum(feature_importances[1]),
#             sum(feature_importances[2]),
#             sum(feature_importances[3]),
#             sum(feature_importances[4]),
#             sum(feature_importances[5]),
#         ]

#         x = np.arange(len(summed_importances))

#         plt.figure()
#         plt.title("Cartpole-v0")
#         plt.bar(x, summed_importances)
#         plt.xticks(
#             x,
#             [
#                 "feature-1",
#                 "feature-2",
#                 "feature-3",
#                 "feature-4",
#                 r"$d_t^{r}$",
#                 r"$d_t^{h}$",
#             ],
#         )
#         plt.savefig("importances_state_" + str(self.testing_state) + ".jpg")
