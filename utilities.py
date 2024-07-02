import gym
import numpy as np
import os
import torch


def check_paths(paths):
    """
    Check the existence of paths.

    @param paths: list of paths to check.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def compute_action_value_parallel(actor_critic_model, obs, device):
    """
    Given an actor critic model and an observation vector, compute actions and values.

    @param actor_critic_model: model.
    @param obs: states vector with observations from each environment.
    @param device: device on which computation is performed.
    @return: actions, action probabilities, values
    """
    actor_critic_model.eval()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action_dist, values = actor_critic_model(obs_tensor)

        actions = torch.multinomial(action_dist, 1)
        action_probs = action_dist.gather(dim=1, index=actions).squeeze(-1)

        actions = actions.squeeze().cpu().numpy()
        action_probs = action_probs.cpu().numpy()
        values = values.squeeze().cpu().numpy()

    return actions, action_probs, values


def make_env(game, seed, num_levels, render_mode="rgb_array", dist_mode="easy", use_background=True):
    """
    Creates a configurable environment for the specified game.

    @param game: The name of the game to be loaded, e.g., 'coinrun'.
    @param seed: An integer to seed the environment for reproducibility.
    @param num_levels: The number of levels to be used within the game; if set to 0, it starts at level 1 otherwise is random.
    @param render_mode: Specifies the mode for rendering outputs ('rgb_array' for image array). Default is 'rgb_array'.
    @param dist_mode: Specifies the distribution mode for the level generation ('easy', 'hard'). Default is 'easy'.
    @param use_background: A boolean to specify whether the background should be used in the environment. Default is True.
    @return: A function that when called, initializes and returns the configured game environment.
    """
    if num_levels == 0:
        start_level = 1
    else:
        start_level = np.random.randint(num_levels)
    def _init():
        env = gym.make(f"procgen:procgen-{game}-v0",
                       start_level=start_level,
                       num_levels=num_levels,
                       distribution_mode=dist_mode,
                       use_backgrounds=use_background,
                       rand_seed=seed,
                       center_agent=True,
                       render_mode=render_mode)
        return env
    return _init


