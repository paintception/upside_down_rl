from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .core import CatchEnv
from .renderer import Renderer


class CatchAdaptor(gym.Env):
    """Adapts the CatchEnv game to the OpenAI Gym interface.

    This class provides a wrapper for the CatchEnv game,
    making it compatible with the Gymnasium environment framework.
    It handles action and observation spacedefinitions, rendering,
    and environment interaction.

    Parameters
    ----------
    render : bool, optional
        If True or "human", renders the environment in a human-viewable window.
        If "rgb_array", renders the environment to an RGB array.
        Default is False.
    numpy_type : str, optional
        The NumPy data type for the observation array. Default is "float32".
    **catch_kwargs
        Additional keyword arguments to pass to the CatchEnv constructor.
    """

    def __init__(
        self, render: bool = False, numpy_type: str = "float32", **catch_kwargs
    ):
        super().__init__()
        self.catch = CatchEnv(**catch_kwargs)
        self.np_type = numpy_type
        self.action_space = spaces.Discrete(3)
        self.obs_shape = (84, 84)
        self.dense = catch_kwargs.get("dense", None)
        if self.dense:
            self.observation_space = spaces.Box(
                np.array([0, 0, 0]), np.array([21, 21, 21]), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.obs_shape, dtype=np.uint8
            )
        self.render_mode = render
        if self.render_mode:
            self.GUI = Renderer(self.obs_shape)

    def step(
        self, action: int
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        Parameters
        ----------
        action : int
            The action to take in the environment
            (0: move left, 1: move right, 2: stay).

        Returns
        -------
        observation : np.ndarray
            The agent's observation of the current environment.
        reward : float
            The amount of reward returned after the previous action.
        terminated : bool
            Whether the episode has ended.
        truncated : bool
            Whether the episode was truncated.
        info : dict
            Contains auxiliary diagnostic information.
        """
        state, reward, done = self.catch.step(action)
        self.state = state
        if self.render_mode:
            self.render()

        # terminated vs truncated: see gymnasium documentation
        # https://gymnasium.farama.org/api/env/
        # in this environment we do not have a difference between the two.
        obs = state
        if not self.dense:
            obs = np.reshape(obs, self.obs_shape).astype(self.np_type)

        return (
            obs,
            reward,
            done,  # terminated
            done,  # trucated
            {},  # empty info
        )

    def reset(self, **_) -> Tuple[NDArray, Dict[str, Any]]:
        """Resets the environment to an initial state and
        returns the initial observation.

        Returns
        -------
        observation : np.ndarray
            The initial observation.
        info : dict
            Contains auxiliary diagnostic information.
        """
        obs = self.catch.reset()
        if not self.dense:
            obs = np.reshape(obs, self.obs_shape)
        return obs, {}

    def render(self):
        """Renders the environment.

        If the 'render' parameter is set,
        this method will display the environment
        either in a human-viewable window or as an RGB array.
        """
        if self.render_mode:
            self.GUI(self.state)

    def close(self):
        """Closes the renderer if it is active."""
        if self.render_mode:
            self.GUI.quit()
