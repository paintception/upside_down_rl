from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize


@dataclass
class CatchEnv:
    """A simple 2D Catch environment for reinforcement learning.

    This environment simulates a game where the agent controls a paddle
    at the bottom of the screen and tries to catch a falling ball.
    The state is represented as an image, and the actions are discrete
    movements of the paddle.

    Attributes
    ----------
    paddle_size: int, default=5
        The size of the paddle in pixels.
    random_background: bool, default=False
        Whether to use a random background image.
    discrete_background: bool, default=False
        If True and random_background is True,
        the background will be chosen from a discrete set of values.
    scale_value: int, default=255
        The scaling factor for the image values.
    """

    paddle_size: int = 5
    random_background: bool = False
    discrete_background: bool = False
    scale_value: int = 255
    dense: bool = False

    size: int = field(init=False, default_factory=lambda: 21)
    scale_factor: int = field(init=False, default_factory=lambda: 4)
    image: np.ndarray = field(init=False)
    background: np.ndarray = field(init=False)
    left_paddle_offset: int = field(init=False)
    right_paddle_offset: int = field(init=False)

    def __post_init__(self):
        """Initializes internal environment variables after object creation."""
        self.final_size = (
            self.size * self.scale_factor,
            self.size * self.scale_factor,
        )
        self.default_size = (self.size, self.size)
        if self.random_background:
            if self.discrete_background:
                self.background = np.random.choice(
                    np.linspace(0, 0.5, 10),
                    size=self.final_size,
                )
            else:
                self.background = resize(
                    np.random.choice(
                        np.linspace(0, 0.999, 10),
                        size=self.default_size,
                    ),
                    self.final_size,
                )

        self.image = np.zeros(self.default_size)
        self.left_paddle_offset = self.paddle_size // 2
        self.right_paddle_offset = self.left_paddle_offset + (
            self.paddle_size % 2
        )

        self.actions = {
            0: lambda self=self: max(self.pos - 2, self.left_paddle_offset),
            1: lambda self=self: min(
                self.pos + 2, self.size - self.right_paddle_offset - 1
            ),
            2: lambda self=self: self.pos,
        }

    def _update_ball(self):
        """Updates the position of the ball in the environment.

        This method updates the ball's position based on its current velocity
        and checks for collisions with the walls.
        If a collision occurs, the ball's velocity is reversed appropriately.
        """
        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size - 1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx -= 2 * self.ballx
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1

    def _update_paddle(self):
        """Updates the position of the paddle in the environment.

        This method clears the previous position of the paddle and
        redraws it at its new position based on the current `self.pos` value.
        """
        self.image[-5].fill(0)
        left_pos = self.pos - self.left_paddle_offset
        right_pos = self.pos + self.right_paddle_offset

        self.image[
            -5,
            left_pos:right_pos,
        ] = np.ones(self.paddle_size)

    def _compute_terminal(self):
        """Determines if the episode is terminal and calculates the reward.

        This method checks if the ball has reached the bottom of the screen,
        indicating the end of an episode. If so, it calculates a reward based
        on whether the ball was caught by the paddle.

        Returns
        -------
        reward : int
            The reward for the current timestep
            (1 if the ball is caught, 0 otherwise).
        terminal : bool
            Whether the episode has ended.
        """
        terminal = self.bally == self.size - 5
        reward = terminal and (
            -self.left_paddle_offset
            <= self.ballx - self.pos
            <= self.right_paddle_offset
        )
        return int(reward), terminal

    def step(self, action: int) -> Tuple[NDArray, int, bool]:
        """Takes a step in the environment.

        Parameters
        ----------
        action: int
            The action to take: 0 (move left), 1 (move right), or 2 (stay).

        Returns
        -------
        image: np.ndarray
            The rendered image of the environment.
        reward: int
            The reward obtained after taking the action.
        terminal: bool
            Whether the episode has ended.
        """
        self.pos = self.actions[action]()
        self._update_ball()
        self._update_paddle()

        image = resize(
            self.image,
            (self.size * self.scale_factor, self.size * self.scale_factor),
        )
        image[image != 0] = 1
        if self.random_background:
            mask = image == 0
            image[mask] = self.background[mask]
        if self.dense:
            return (
                [self.ballx, self.bally, self.pos],
                *self._compute_terminal(),
            )
        return (image * self.scale_value, *self._compute_terminal())

    def reset(self) -> NDArray:
        """Resets the environment to its initial state.

        Returns
        -------
        image: np.ndarray
            The initial rendered image of the environment.
        """
        self.image = np.zeros((self.size, self.size))
        self.pos = np.random.randint(
            self.left_paddle_offset, self.size - self.right_paddle_offset
        )
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(self.size), 4
        self.image[self.bally, self.ballx] = 1
        self._update_paddle()

        return self.step(2)[0]
