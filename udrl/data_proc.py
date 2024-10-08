from pathlib import Path
import numpy as np
import json
import csv

naming = {
    "neural": "NN",
    "ensemble.ExtraTreesClassifier": "ET",
    "ensemble.RandomForestClassifier": "RF",
}

if __name__ == "__main__":
    path = Path("data")
    csvs_path = path / "csvs"
    csvs_path.mkdir(parents=True, exist_ok=True)
    for env in path.iterdir():
        all_paths = list(set([p.parent for p in env.rglob("*.npy")]))
        if not all_paths:
            continue
        toy_rewards = np.load(all_paths[0] / "train_rewards.npy")
        data = {"episode": list(range(len(toy_rewards)))}
        estimators = {
            "neural": ([], [], [], []),
            "ensemble.ExtraTreesClassifier": ([], [], [], []),
            "ensemble.RandomForestClassifier": ([], [], [], []),
        }
        for exp in all_paths:
            print(exp)
            rewards = np.load(exp / "train_rewards.npy")

            with open((exp / "conf.json"), "r") as f:
                conf = json.load(f)

            estimators[conf["estimator_name"]][0].append(list(rewards[:, 0]))
            estimators[conf["estimator_name"]][1].append(list(rewards[:, 1]))
            estimators[conf["estimator_name"]][2].append(conf["test_mean"])
            estimators[conf["estimator_name"]][3].append(conf["test_std"])

        for k, v in estimators.items():
            data[naming[k] + "_mean"] = [
                "{:.2f}".format(np.mean(x)) for x in zip(*v[0])
            ]
            data[naming[k] + "_std"] = [
                "{:.2f}".format(np.std(x)) for x in zip(*v[0])
            ]
            print(f"{k}:{env.name}-> {np.median(v[2])} +- {np.median(v[3])}")

        with open(csvs_path / f"{env.name}.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(data.keys())
            w.writerows(zip(*data.values()))
