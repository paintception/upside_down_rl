from .adptor import CatchAdaptor
from .core import CatchEnv

env_names = [
    "base",
    "small_paddle",
    "random_background",
    "hardest",
    "discrete_background",
]

env_names = ["catch_" + x for x in env_names]


def make_catch_conf(env_name: str):
    base_args = {
        "random_background": False,
        "discrete_background": False,
        "paddle_size": 5,
    }
    match env_name:
        case "catch_small_paddle":
            base_args["paddle_size"] = 2
        case "catch_discrete_background":
            base_args["random_background"] = True
            base_args["discrete_background"] = True
        case "catch_random_background":
            base_args["random_background"] = True
        case "catch_hardest":
            base_args["random_background"] = True
            base_args["paddle_size"] = 2
    return base_args


__all__ = ["CatchEnv", "CatchAdaptor", "make_catch_conf", "env_names"]
