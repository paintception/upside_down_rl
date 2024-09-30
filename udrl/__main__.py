from udrl.agent import UpsideDownAgent, AgentHyper
from udrl.policies import SklearnPolicy, NeuralPolicy
from udrl.catch import CatchAdaptor
from dataclasses import dataclass, asdict
import gymnasium as gym
from tqdm import trange
import numpy as np
import warnings
import argparse
from udrl.cli import (
    with_meta,
    create_argparse_dict,
    create_experiment_from_args,
    dataclass_non_defaults_to_string,
    apply,
)
from pathlib import Path
import json
import torch
import random as rnd


@dataclass
class UDRLExperiment:
    """Configuration for an Upside-Down Reinforcement Learning experiment."""

    env_name: str = with_meta(
        "CartPole-v0", "Name of the Gym environment to use "
    )
    estimator_name: str = with_meta(
        "ensemble.RandomForestClassifier",
        "neural for the NN or a fully qualified name of the "
        "scikit-learn estimator class "
        "for the policy",
    )
    seed: int = with_meta(42, "Random seed for reproducibility")

    max_episode: int = with_meta(500, "Maximum number of training episodes ")
    collect_iter: int = with_meta(
        15, "Number of episodes to collect between training steps "
    )
    train_per_iter: int = with_meta(
        100, "Number of train iteration for each collected episode "
    )
    batch_size: int = with_meta(
        0,
        "Batch size for training the policy."
        "If batch_size <= 0, use the entire replay buffer",
    )

    warm_up: int = with_meta(
        50, "Number of initial random episodes to populate the replay buffer"
    )
    memory_size: int = with_meta(700, "Maximum size of the replay buffer")
    last_few: int = with_meta(
        75,
        "Number of recent episodes to consider for exploratory command sampling",
    )
    testing_period: int = with_meta(
        10, "After how many training loop we perform the testing of the agent"
    )

    horizon_scale: float = with_meta(
        0.02, "Scaling factor for desired horizon in commands "
    )
    return_scale: float = with_meta(
        0.02, "Scaling factor for desired return in commands"
    )

    epsilon: float = with_meta(
        0.2, "Exploration rate for epsilon-greedy action selection"
    )
    save_desired: bool = with_meta(
        False, "Save desired_horizon and desired_return during training"
    )

    final_testing: bool = with_meta(
        True, "Whether to perform final testing after training "
    )
    final_testing_sample: int = with_meta(
        100, "Number of episodes to evaluate during final testing "
    )
    final_desired_return: int = with_meta(
        200, "Desired return for final testing episodes"
    )
    final_desired_horizon: int = with_meta(
        200, "Desired horizon for final testing episodes "
    )
    save_policy: bool = with_meta(True, "Whether to save the trained policy ")
    save_learning_infos: bool = with_meta(
        True, "Whether to save the learning infos"
    )


def dump_dict(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def run_experiment(conf: UDRLExperiment):
    """Runs an Upside-Down Reinforcement Learning experiment.

    Parameters
    ----------
    conf : UDRLExperiment
        Configuration for the experiment.

    Returns
    -------
    None

    Notes
    -----
    * Trains an agent using the specified policy and environment.
    * Collects episodes of experience and updates the policy.
    * Optionally performs final testing,saves the policy and learning infos.
    """
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    rnd.seed(conf.seed)

    toy_env = (
        CatchAdaptor(dense=True)
        if conf.env_name == "catch"
        else gym.make(conf.env_name)
    )
    if conf.estimator_name == "neural":
        policy = NeuralPolicy(
            toy_env.observation_space.shape[0],
            action_size=toy_env.action_space.n,
        )
    else:
        policy = SklearnPolicy(
            epsilon=conf.epsilon,
            estimator_name=conf.estimator_name,
            action_size=toy_env.action_space.n,
        )
    agent = UpsideDownAgent(
        conf=apply(AgentHyper, asdict(conf)),
        policy=policy,
    )
    epi_bar = trange(conf.max_episode)

    returns = []
    test_returns = []
    infos = []
    desired_returns = []
    desired_horizons = []
    test_reward_mean = 0
    test_reward_std = 0
    for e in epi_bar:
        metric = []
        for _ in range(conf.train_per_iter):
            info = agent.train()
            metric.append(info["metric"])
            infos.append(info)

        episodic_rewards = []
        for _ in range(conf.collect_iter):
            r, dr, dh = agent.collect_episode(
                *agent.sample_exploratory_commands()
            )
            episodic_rewards.append(r)
            desired_returns.extend(dr)
            desired_horizons.extend(dh)

        ep_r_mean = np.mean(episodic_rewards)
        ep_r_std = np.std(episodic_rewards)
        returns.append((ep_r_mean, ep_r_std))

        if e % conf.testing_period == 0:
            test_reward = [
                agent.collect_episode(
                    conf.final_desired_return,
                    conf.final_desired_horizon,
                    test=True,
                    store_episode=False,
                )[0]
                for _ in range(conf.final_testing_sample)
            ]
            test_reward_mean = np.mean(test_reward)
            test_reward_std = np.std(test_reward)
            test_returns.append((test_reward_mean, test_reward_std))

        epi_bar.set_postfix(
            {
                "mean": test_reward_mean,
                "std": test_reward_std,
                "mean_m": np.mean(metric),
                "std_m": np.std(metric),
            }
        )

    exp_name = dataclass_non_defaults_to_string(conf)
    base_path = Path("data") / conf.env_name / exp_name / str(conf.seed)
    base_path.mkdir(parents=True, exist_ok=True)
    final_res = {}
    if conf.final_testing:
        print("Start Testing...")
        final_r = [
            agent.collect_episode(
                conf.final_desired_return,
                conf.final_desired_horizon,
                test=True,
                store_episode=False,
            )[0]
            for _ in trange(conf.final_testing_sample)
        ]
        final_res["test_mean"] = np.mean(final_r)
        final_res["test_std"] = np.std(final_r)
        print(f"Final result:\n{np.mean(final_r)} +- {np.std(final_r)}")

    dump_dict(asdict(conf) | final_res, str(base_path / "conf.json"))
    if conf.save_policy:
        agent.policy.save(str(base_path / "policy"))

    if conf.save_learning_infos:
        np.save(str(base_path / "train_rewards.npy"), returns)
        np.save(str(base_path / "test_rewards.npy"), test_returns)
        np.save(str(base_path / "desired_returns.npy"), desired_returns)
        np.save(str(base_path / "desired_horizons.npy"), desired_horizons)
        dump_dict(infos, str(base_path / "learning_infos.json"))


warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
parser = argparse.ArgumentParser(
    description="Runs an Upside-Down Reinforcement Learning experiment."
    "NOTE: Default values are for the CartPole env with RandomForestClassifier"
)
arguments = create_argparse_dict(UDRLExperiment)
for k, v in arguments.items():
    parser.add_argument(k, **v)
args = parser.parse_args()
conf = create_experiment_from_args(args, UDRLExperiment)
print(conf)

run_experiment(conf)
