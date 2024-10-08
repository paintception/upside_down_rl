import gymnasium as gym
import pygame
import numpy as np
from .policies import SklearnPolicy
from .agent import UpsideDownAgent, AgentHyper
from pathlib import Path
import json


def normalize_value(value, is_bounded, low=None, high=None):
    return (value - low) / (high - low)


def draw_bar(screen, start, value, max_length, color, height=20, mid=True):
    bar_length = value * max_length
    pygame.draw.rect(screen, color, (*start, bar_length, height))
    pygame.draw.rect(screen, (0, 0, 0), (*start, max_length, height), 2)
    if mid:
        mid_x = start[0] + max_length / 2
        pygame.draw.line(
            screen, (0, 0, 0), (mid_x, start[1]), (mid_x, start[1] + height), 2
        )


def create_button(text, position, size):
    font = pygame.font.Font(None, 36)
    button_rect = pygame.Rect(position, size)
    text_surf = font.render(text, True, (0, 0, 0))
    text_rect = text_surf.get_rect(center=button_rect.center)
    return button_rect, text_surf, text_rect


def visualize_environment(
    screen,
    state,
    env,
    env_surface,
    paused,
    feature_importances,
    epoch,
    max_epoch=200,
):
    screen_width, screen_height = screen.get_size()
    screen.fill((255, 255, 255))
    screen.blit(env_surface, (0, 0))

    num_dims = len(feature_importances)
    bar_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 128),  # Gray
        (192, 192, 192),  # Light Gray
        (255, 165, 0),  # Orange
        (255, 192, 203),  # Pink
    ]

    bar_starts = [
        (screen_width - 350, 50 + i * 70) for i in range(num_dims + 1)
    ]
    max_length = 300

    for i, (start, color) in enumerate(zip(bar_starts, bar_colors)):
        if i == 0:
            normalized_value = epoch / max_epoch
            draw_bar(
                screen, start, normalized_value, max_length, color, mid=False
            )
            font = pygame.font.Font(None, 30)
            text = font.render(
                f"Epoch {epoch}",
                True,
                (0, 0, 0),
            )
            screen.blit(text, (start[0], start[1] - 30))
            continue
        i -= 1
        normalized_value = feature_importances[i] / 30
        draw_bar(screen, start, normalized_value, max_length, color)

        font = pygame.font.Font(None, 30)
        what = f"Importance  {i}"
        if len(feature_importances) - i == 2:
            what = "Desired Return"
        if len(feature_importances) - i == 1:
            what = "Desired Horizon"

        text = font.render(
            f"{what}: {feature_importances[i]:.2f}", True, (0, 0, 0)
        )
        screen.blit(text, (start[0], start[1] - 30))

        desc = "(Range: 0 to 30)"
        desc_text = pygame.font.Font(None, 24).render(
            desc, True, (100, 100, 100)
        )
        screen.blit(desc_text, (start[0], start[1] + 25))

    button_width, button_height = 100, 50
    reset_button, reset_text, reset_text_rect = create_button(
        "Reset", (10, screen_height - 60), (button_width, button_height)
    )
    pause_play_button, pause_play_text, pause_play_text_rect = create_button(
        "Pause" if not paused else "Play",
        (120, screen_height - 60),
        (button_width, button_height),
    )
    next_button, next_text, next_text_rect = create_button(
        "Next", (230, screen_height - 60), (button_width, button_height)
    )
    save_button, save_text, save_text_rect = create_button(
        "Save", (340, screen_height - 60), (button_width, button_height)
    )

    pygame.draw.rect(screen, (200, 200, 200), reset_button)
    pygame.draw.rect(screen, (200, 200, 200), pause_play_button)
    pygame.draw.rect(screen, (200, 200, 200), next_button)
    pygame.draw.rect(screen, (200, 200, 200), save_button)
    screen.blit(reset_text, reset_text_rect)
    screen.blit(pause_play_text, pause_play_text_rect)
    screen.blit(next_text, next_text_rect)
    screen.blit(save_text, save_text_rect)

    pygame.display.flip()
    return reset_button, pause_play_button, next_button, save_button


def run_visualization(
    env_name,
    agent,
    init_desired_return,
    init_desired_horizon,
    max_epoch,
    base_path,
):
    base_path = (
        Path(base_path) / env_name / agent.policy.estimator.__str__()[:-2]
    )
    base_path.mkdir(parents=True, exist_ok=True)
    desired_return = init_desired_return
    desired_horizon = init_desired_horizon

    pygame.init()
    screen = pygame.display.set_mode((1000, 800))
    pygame.display.set_caption(f"{env_name} Visualization")

    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()

    clock = pygame.time.Clock()
    epoch = 0
    save_index = 0

    running = True
    paused = False
    step = False
    while running:

        env_render = env.render()
        env_surface = pygame.surfarray.make_surface(env_render.swapaxes(0, 1))
        if not paused or step:
            command = np.array(
                [
                    desired_return * agent.conf.return_scale,
                    desired_horizon * agent.conf.horizon_scale,
                ]
            )
            command = np.expand_dims(command, axis=0)
            state = np.expand_dims(state, axis=0)

            action = agent.policy(state, command, True)

            ext_state = np.concatenate((state, command), axis=1)

            state, reward, done, truncated, info = env.step(action)

            feature_importances = {
                idx: [] for idx in range(ext_state.shape[1])
            }

            for t in agent.policy.estimator.estimators_:
                branch = np.array(
                    t.decision_path(ext_state).todense(), dtype=bool
                )
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
                sum(feature_importances.get(k, [0.001]))
                for k in range(len(feature_importances.keys()))
            ]

            epoch += 1

        reset_button, pause_play_button, next_button, save_button = (
            visualize_environment(
                screen,
                state,
                env,
                env_surface,
                paused,
                summed_importances,
                epoch,
                max_epoch,
            )
        )

        if done or truncated:
            state, _ = env.reset()
            desired_horizon = init_desired_horizon
            desired_return = init_desired_return
            epoch = 0

        step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if reset_button.collidepoint(event.pos):
                    state, _ = env.reset()

                    desired_horizon = init_desired_horizon
                    desired_return = init_desired_return
                    epoch = 0
                elif pause_play_button.collidepoint(event.pos):
                    paused = not paused
                elif (
                    next_button.collidepoint(event.pos) and paused
                ):  # Only when paused
                    step = True
                elif save_button.collidepoint(event.pos):
                    pygame.image.save(
                        env_surface,
                        str(base_path / f"env_image_{save_index}.png"),
                    )
                    with open(
                        str(base_path / f"info_{save_index}.json"), "w"
                    ) as f:
                        json.dump(
                            {
                                "state": {
                                    i: str(val) for i, val in enumerate(state)
                                },
                                "feature": {
                                    i: str(val)
                                    for i, val in enumerate(summed_importances)
                                },
                                "action": str(action),
                                "reward": str(reward),
                                "desired_return": str(desired_return + reward),
                                "desired_horizon": str(desired_horizon + 1),
                            },
                            f,
                            indent=4,
                        )

                    save_index += 1
        clock.tick(5)

    env.close()
    pygame.quit()


# LunarLander-v2:RT:43:r.57:h.102 -> -92.03 +- 81.51,max 36.37,min -327.94
# Acrobot-v1:RT:44:r.-79:h.82 -> -79.00 +- 47.01,max -64.00,min -500.00


base_path = Path("data")
# env = "CartPole-v0"
env = "Acrobot-v1"
# env = "LunarLander-v2"
estimator = "RandomForestClassifier"
seed = str(44)
conf_name = "train_per_iter1"
desired_return = -79
desired_horizon = 82
max_epoch = 500

path = base_path / env / conf_name / seed

policy = SklearnPolicy.load(str(path / "policy"))
hyper = AgentHyper(
    env,
    warm_up=0,
    # horizon_scale=horizon_scale,
    # return_scale=return_scale,
)

agent = UpsideDownAgent(hyper, policy)

run_visualization(
    env, agent, desired_return, desired_horizon, max_epoch, "data/viz_examples"
)
