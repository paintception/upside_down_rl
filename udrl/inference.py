import matplotlib.pyplot as plt
import numpy as np
from .policies import SklearnPolicy, NeuralPolicy
from .agent import UpsideDownAgent, AgentHyper
from pathlib import Path
from collections import Counter
from tqdm import trange


def get_common(base, env, conf, seed):

    path = base / env / conf / seed

    if not path.exists():
        print("Cannot find path")
        return None, None
    algo_name = (
        "NN" if "neural" in conf else ("ET" if "Extra" in conf else "RT")
    )

    des_ret = np.load(str(path / "desired_returns.npy")).astype(int)
    des_hor = np.load(str(path / "desired_horizons.npy")).astype(int)
    # rew = np.load(str(path / "train_rewards.npy")).astype(int)[:, 0]

    te = []
    prev = -np.inf
    for i, x in enumerate(des_hor):
        if prev < x:
            te.append(i)
        prev = x

    init_des_ret = des_ret[te]
    init_des_hor = des_hor[te]

    mean_des_ret = []
    mean_des_hor = []
    tmp_r = []

    tmp_h = []
    for i, (ret, hor) in enumerate(zip(init_des_ret, init_des_hor)):
        tmp_r.append(ret)
        tmp_h.append(hor)
        if i % 15 == 0:
            mean_des_hor.append(np.mean(tmp_h))
            mean_des_ret.append(np.mean(tmp_r))
            tmp_r = []
            tmp_h = []

    common_hor = Counter(init_des_hor[-1500:]).most_common()[0][0]
    common_ret = Counter(init_des_ret[-1500:]).most_common()[0][0]
    print(f"{env}:{algo_name}.horizon-> {common_hor}")
    print(f"{env}:{algo_name}.return-> {common_ret}")
    return common_ret, common_hor


def test_desired(base, env, conf, des_ret, des_hor):

    algo_name = (
        "NN" if "neural" in conf else ("ET" if "Extra" in conf else "RT")
    )
    if des_hor is None or des_ret is None:
        print(f"Invalid desired for {env}:{algo_name}")
        return
    for path in (base / env / conf).iterdir():
        if "neural" in conf:
            policy = NeuralPolicy.load(str(path / "policy"))
        else:
            policy = SklearnPolicy.load(str(path / "policy"))

        hyper = AgentHyper(env, warm_up=0)

        agent = UpsideDownAgent(hyper, policy)

        final_r = [
            agent.collect_episode(
                des_ret,
                des_hor,
                test=True,
                store_episode=False,
            )[0]
            for _ in range(100)
        ]
        print(
            f"{env}:{algo_name}:{path.name}:r.{des_ret}:h.{des_hor}"
            f" -> {np.median(final_r):.2f} +- {np.std(final_r):.2f}"
            f",max {np.max(final_r):.2f},min {np.min(final_r):.2f}"
        )


base = Path("/home/vimmoos/upside_down_rl/data")
confs = {
    "NN": "estimator_nameneural_batch_size256_warm_up260",
    "ET": "estimator_nameensemble.ExtraTreesClassifier_train_per_iter1",
    "RT": "train_per_iter1",
}
envs = ["LunarLander-v2", "Acrobot-v1"]
seeds = [str(45), str(46)]

res = {}


for env in envs:
    res[env] = {}
    for algo_name, conf in confs.items():
        res[env][algo_name] = {}
        for seed in seeds:
            ret, hor = get_common(base, env, conf + "_save_desiredTrue", seed)
            res[env][algo_name][seed] = (ret, hor)


pprint(res)

for env, algos in res.items():
    for algo, seeds in algos.items():
        for _, vals in seeds.items():
            test_desired(base, env, confs[algo], *vals)


# plt.plot(mean_des_ret)
# plt.plot(mean_des_hor)
# plt.plot(rew)
# plt.show()
