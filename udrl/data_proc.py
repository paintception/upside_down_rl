from pathlib import Path
import numpy as np
import json
import csv

if __name__ == "__main__":
    path = Path("data")
    csvs_path = path / "csvs"
    csvs_path.mkdir(parents=True, exist_ok=True)
    for env in path.iterdir():
        all_paths = [p.parent for p in env.rglob("*.npy")]
        if not all_paths:
            continue
        toy_rewards = np.load(all_paths[0] / "rewards.npy")
        print
        data = {"episode": list(range(len(toy_rewards)))}
        for exp in all_paths:
            rewards = np.load(exp / "rewards.npy")

            with open((exp / "conf.json"), "r") as f:
                conf = json.load(f)

            data[conf["estimator_name"] + "_mean"] = list(rewards[:, 0])
            data[conf["estimator_name"] + "_std"] = list(rewards[:, 1])

        with open(csvs_path / f"{env.name}.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(data.keys())
            w.writerows(zip(*data.values()))
