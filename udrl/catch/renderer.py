from dataclasses import dataclass, field
from os import environ
from typing import Tuple

import pygame
from numpy.typing import NDArray

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


@dataclass
class Renderer:
    """A renderer for visualizing the CatchEnv game using Pygame.

    This class initializes a Pygame screen and provides a method to render the
    CatchEnv game state as an image onto the screen.

    Attributes
    ----------
    size : Tuple[int, int]
        The size of the environment to render (height, width).
    scale_factor : int, default=5
        The scaling factor for the rendered image.
    """

    size: Tuple[int, int]
    scale_factor: int = 5

    screen: pygame.surface.Surface = field(init=False)

    def __post_init__(self):
        """Initializes the Pygame display after object creation."""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                self.size[1] * self.scale_factor,
                self.size[0] * self.scale_factor,
            )
        )

    def quit(self):
        """Quits the Pygame display."""
        pygame.quit()

    def __call__(self, image: NDArray):
        """Renders the CatchEnv game state onto the Pygame screen.

        Parameters
        ----------
        image : np.ndarray
            A 2D NumPy array representing the game state. The array should have
            values that correspond to pixel intensities or colors.
        """

        scaled_size = (
            image.shape[0] * self.scale_factor,
            image.shape[1] * self.scale_factor,
        )
        scaled_image = pygame.transform.scale(
            pygame.surfarray.make_surface(image.T), scaled_size
        )

        # Blit (copy) the scaled image onto the screen
        self.screen.blit(scaled_image, (0, 0))
        pygame.display.flip()
