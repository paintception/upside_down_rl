from udrl.agent import UpsideDownAgent, AgentHyper
from udrl.policies import SklearnPolicy
from udrl import utils
from tqdm import trange
import numpy as np
import warnings


def run_experiment():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--approximator", type=str, default="forest")
    parser.add_argument("--environment", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    approximator = args.approximator
    environment = args.environment
    seed = args.seed
    print(args)

    episodes = 10
    collect_episode = 15
    returns = []
    policy = SklearnPolicy(0.2, "ensemble.RandomForestClassifier")
    agent = UpsideDownAgent(AgentHyper(environment, batch_size=0), policy)
    epi_bar = trange(episodes)

    for e in epi_bar:
        agent.train()

        tmp_r = []
        for i in range(collect_episode):
            # Line 5 Algorithm 1
            r = agent.collect_episode(*agent.sample_exploratory_commands())
            tmp_r.append(r)

        epi_bar.set_postfix(
            {
                "mean": np.mean(tmp_r),
                "std": np.std(tmp_r),
            }
        )
        # print()
        returns.append(np.mean(tmp_r))

    final_r = agent.collect_episode(200, 200, test=True)
    print(f"final result:{final_r}")

    agent.policy.save("test_policy")

    # utils.save_results(environment, approximator, seed, returns)

    # if approximator == "neural_network":
    #     utils.save_trained_model(environment, seed, agent.behaviour_function)


warnings.simplefilter("ignore", DeprecationWarning)
run_experiment()


# pol = SklearnPolicy.load("test_policy")

# agent = UpsideDownAgent(AgentHyper("CartPole-v0", batch_size=0), pol)

# agent.collect_episode(200,200,test=True)
