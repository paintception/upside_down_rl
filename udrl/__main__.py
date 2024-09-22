import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from udrl.agent import UpsideDownAgent, AgentHyper
from udrl.policies import SklearnPolicy, NeuralPolicy
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
    collect_episode: int = with_meta(
        15, "Number of episodes to collect between training steps "
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

    horizon_scale: float = with_meta(
        0.02, "Scaling factor for desired horizon in commands "
    )
    return_scale: float = with_meta(
        0.02, "Scaling factor for desired return in commands"
    )

    epsilon: float = with_meta(
        0.2, "Exploration rate for epsilon-greedy action selection "
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
    save_learning_rewards: bool = with_meta(
        True, "Whether to save the learning rewards during training"
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
    * Optionally performs final testing,saves the policy and learning rewards.
    """
    toy_env = gym.make(conf.env_name)
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
    for e in epi_bar:
        agent.train()
        episodic_rewards = [
            agent.collect_episode(*agent.sample_exploratory_commands())
            for _ in range(conf.collect_episode)
        ]
        ep_r_mean = np.mean(episodic_rewards)
        ep_r_std = np.std(episodic_rewards)
        epi_bar.set_postfix({"mean": ep_r_mean, "std": ep_r_std})
        returns.append((ep_r_mean, ep_r_std))

    exp_name = dataclass_non_defaults_to_string(conf)
    base_path = Path("data") / conf.env_name / str(conf.seed) / exp_name
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
            )
            for _ in trange(conf.final_testing_sample)
        ]
        final_res["test_mean"] = np.mean(final_r)
        final_res["test_std"] = np.std(final_r)
        print(f"Final result:\n{np.mean(final_r)} +- {np.std(final_r)}")

    dump_dict(asdict(conf) | final_res, str(base_path / "conf.json"))
    if conf.save_policy:
        agent.policy.save(str(base_path / "policy"))

    if conf.save_learning_rewards:
        np.save(str(base_path / "rewards.npy"), returns)


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
