import warnings
from stable_baselines3.common.vec_env import SubprocVecEnv
from utilities import *
from plot_and_save_parallel import LearningInfo
from model import ImpalaModel

# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NUM_ENVS = 1
NUM_LEVELS = 0
SEED = 1111

MAX_TIMESTEPS = 50000
PLOT_EVERY = 25000

RANDOM = False

GAME = "fruitbot"
MODEL_WEIGHTS = f"weights/{GAME}/model_weights.pt"
PLOT_PATH = f"test_plots/{GAME}"

if __name__ == '__main__':
    if not RANDOM:
        check_paths([PLOT_PATH])
    env = SubprocVecEnv([make_env(GAME, SEED+1, NUM_LEVELS, "human")])

    actor_critic_model = ImpalaModel().to("cpu")
    actor_critic_model.load_state_dict(torch.load(MODEL_WEIGHTS))

    learningInfo = LearningInfo(timestep_to_plot=PLOT_EVERY,
                                plot_save_path=PLOT_PATH,
                                n_env=NUM_ENVS,
                                game=GAME)

    print("\nTesting started\n")
    total_timesteps = 0
    env_timesteps = np.zeros(NUM_ENVS)
    obs = env.reset()
    done = False

    # number of episodes for each environment
    n_episodes = np.zeros(NUM_ENVS, dtype=int)

    # loop until MAX_TIMESTEPS is reached
    while total_timesteps < MAX_TIMESTEPS:
        if RANDOM:
            action = env.action_space.sample()
        else:
            action, _, _ = compute_action_value_parallel(actor_critic_model, obs, "cpu")

        next_obs, rewards, done, _ = env.step([action])
        total_timesteps += NUM_ENVS

        env_timesteps += 1
        learningInfo.update(total_timesteps, n_episodes, rewards)

        obs = next_obs

    env.close()
