from .policies import SklearnPolicy
from .agent import UpsideDownAgent, AgentHyper
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest, tee
from tqdm import tqdm, trange
import imageio


def calculate_ep_feat_importance(
    episode, agent, desired_return, desired_horizon
):
    ep_features = []

    for state, _, reward in zip(*episode.values()):
        command = np.array(
            [
                desired_return * agent.conf.return_scale,
                desired_horizon * agent.conf.horizon_scale,
            ]
        )
        command = np.expand_dims(command, axis=0)
        ext_state = np.concatenate((state, command), axis=1)

        feature_importances = {}

        for t in agent.policy.estimator.estimators_:
            branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
            imp = t.tree_.impurity[branch[0]]
            for f, i in zip(
                t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]
            ):
                feature_importances.setdefault(f, []).append(i)

        # Line 8 Algorithm 2
        desired_return -= reward
        # Line 9 Algorithm 2
        desired_horizon = max(desired_horizon - 1, 1)

        summed_importances = [
            sum(feature_importances[k])
            for k in range(len(feature_importances.keys()))
        ]
        ep_features.append(summed_importances)
    return ep_features


def summarize_episodes_feat(
    episodes_feat, summarize_funs: list = [np.mean, np.std]
):
    return [
        [
            [
                fun(list(data))
                for fun, data in zip(
                    summarize_funs,
                    tee(
                        (s for s in state if s is not None),
                        len(summarize_funs),
                    ),
                )
            ]
            for state in zip_longest(*ep)
        ]
        for ep in zip_longest(*episodes_feat, fillvalue=[])
    ]


def calculate_features_importance(
    path: Path,
    env: str,
    desired_return: int,
    desired_horizon: int,
    horizon_scale: float,
    return_scale: float,
    redundancy: int = 100,
):
    policy = SklearnPolicy.load(str(path / "policy"))
    hyper = AgentHyper(
        env,
        warm_up=0,
        horizon_scale=horizon_scale,
        return_scale=return_scale,
    )

    agent = UpsideDownAgent(hyper, policy)

    for _ in trange(redundancy, desc="Collect Data"):
        agent.collect_episode(desired_return, desired_horizon, test=True)

    episodes = [
        {k: v for k, v in ep.items() if k != "summed_rewards"}
        for ep in agent.memory.buffer
    ]

    episodes_feat = [
        calculate_ep_feat_importance(
            ep, agent, desired_return, desired_horizon
        )
        for ep in tqdm(episodes, desc="Calculate importance features")
    ]

    feature_importances = summarize_episodes_feat(episodes_feat)
    return feature_importances


def example_plot(feature_importances):
    for idx, state_feat in tqdm(
        enumerate(feature_importances),
        desc="Plotting",
        total=len(feature_importances),
    ):
        x = np.arange(len(state_feat))

        plt.figure()
        plt.title(f"Cartpole-v0 State {idx}")
        plt.bar(x, [x[0] for x in state_feat], yerr=[x[1] for x in state_feat])

        plt.xticks(
            x,
            [
                *[f"feature-{index}" for index in range(len(state_feat) - 2)],
                r"$d_t^{r}$",
                r"$d_t^{h}$",
            ],
        )
        plt.savefig(f"data/example_plot2/importances_state_{idx}")
        plt.close()


def create_gif_from_plots(
    image_filenames, output_filename="animation.gif", duration=0.5
):
    """Creates a GIF from a list of image filenames."""

    images = [imageio.imread(filename) for filename in image_filenames]
    imageio.mimsave(output_filename, images, duration=duration)


base_path = Path("data")
env = "CartPole-v0"
estimator = "ExtraTreesClassifier"
seed = str(42)
conf_name = "estimator_nameensemble.ExtraTreesClassifier_train_per_iter1"
desired_return = 200
desired_horizon = 200

path = base_path / env / conf_name / seed


res = calculate_features_importance(
    path, env, desired_return, desired_horizon, 0.02, 0.02
)
example_plot(res)

image_filenames = [
    f"data/example_plot2/importances_state_{idx}.png"
    for idx in range(len(res))
]

create_gif_from_plots(image_filenames)


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, HDBSCAN
# from sklearn.decomposition import PCA

# # Assuming you have your data in a numpy array 'data'
# data = np.array(res)[:, :, 0]

# # 1. Apply K-Means clustering
# kmeans = HDBSCAN()
# kmeans.fit(data)
# labels = kmeans.labels_

# # 2. Dimensionality Reduction for visualization (PCA)
# pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
# data_pca = pca.fit_transform(data)

# # 3. Plotting
# plt.figure(figsize=(10, 8))
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap="viridis")
# plt.title("K-Means Clustering Visualization")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.colorbar()
# plt.show()
