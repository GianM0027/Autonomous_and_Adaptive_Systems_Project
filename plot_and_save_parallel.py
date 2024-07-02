import os.path
import numpy as np
from matplotlib import pyplot as plt
import torch

class LearningInfo:
    """
    Class to manage learning information, perform plotting, and handle model weight saving.

    @param timestep_to_plot: Number of timesteps after which to plot metrics.
    @param n_env: Number of parallel environments.
    @param plot_save_path: Path to save plots, defaults to None.
    @param game: Name of the game or environment, defaults to empty string.
    """
    def __init__(self, timestep_to_plot, n_env, plot_save_path=None, game=""):
        self.timestep_to_plot = timestep_to_plot
        self.plot_save_path = plot_save_path
        self.game = game
        self.n_env = n_env
        self.best_score = -np.inf

        # Dictionary to track rewards and episodes lengths for each environment and episode
        self.rewards = {env: {} for env in range(n_env)}
        self.episodes_length = {env: {} for env in range(n_env)}

        # timesteps and episodes trackers
        self.timesteps = 0
        self.n_episodes = 0

    def update(self, timesteps, episodes, rewards, verbose=True):
        """
        Update learning information with new data and possibly trigger plotting of metrics.

        @param timesteps: Total timesteps passed from beginning of training.
        @param episodes: Number of episodes for each environment.
        @param rewards: List of rewards for each environment.
        @param verbose: Flag of plotting verbosity, defaults to True.
        """
        self.timesteps = timesteps
        self.n_episodes = min(episodes)

        for env in range(self.n_env):
            env_episodes = episodes[env]

            # if new episode just started for env, initialize a new bin for rewards and episode length
            if env_episodes not in self.rewards[env]:
                self.rewards[env][env_episodes] = 0
                self.episodes_length[env][env_episodes] = 0

            # append collected data
            self.rewards[env][env_episodes] += rewards[env]
            self.episodes_length[env][env_episodes] += 1

        if (self.timesteps % self.timestep_to_plot == 0) and self.n_episodes > 0:
            self.plot_metrics(verbose)

    def save_weights(self, model, weights_path, early_stopping=False, file_name="model_weights.pt"):
        """
        Save model weights.

        @param model: Model whose weights to save.
        @param weights_path: Path to save weights.
        @param early_stopping: Use early stopping to save only improved models, defaults to False.
        @param file_name: File name to save the model under, defaults to "model_weights.pt".
        """
        model_path = os.path.join(weights_path, file_name)
        if early_stopping:
            if self.n_episodes > 1:
                last_avg_rewards = np.mean([self.rewards[env][self.n_episodes-1] for env in range(self.n_env)])

                if last_avg_rewards > self.best_score:
                    self.best_score = last_avg_rewards
                    torch.save(model.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

    def plot_metrics(self, verbose=True):
        """
        Plot the average sum of rewards within episodes and the average episode lengths across the environments.

        @param verbose: If True, displays the plots.
        """
        # Averages, min and max rewards
        avg_rewards = [np.mean([self.rewards[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]
        min_rewards = [min([self.rewards[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]
        max_rewards = [max([self.rewards[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]

        # Averages, min and max episode lengths
        avg_lengths = [np.mean([self.episodes_length[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]
        min_lengths = [min([self.episodes_length[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]
        max_lengths = [max([self.episodes_length[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]

        plt.figure(figsize=(20, 5))

        # Subplot 1: Plot for Rewards
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.title(f"Rewards Over Episodes at Timestep {self.timesteps}")
        plt.plot(range(self.n_episodes), avg_rewards, 'r')
        plt.fill_between(range(self.n_episodes), min_rewards, max_rewards, color='r', alpha=0.2)
        plt.grid(True, alpha=0.3)

        # Subplot 2: Plot for Episode Lengths
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.title(f"Episode Lengths Over Episodes at Timestep {self.timesteps}")
        plt.plot(range(self.n_episodes), avg_lengths, 'b')
        plt.fill_between(range(self.n_episodes), min_lengths, max_lengths, color='b', alpha=0.2)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.plot_save_path:
            plt.savefig(f"{self.plot_save_path}/metrics_plot_{self.timesteps}.png")

        if verbose:
            plt.show()

        plt.close()

    def summary(self, dataset="train", percentage=0.03):
        """
        Print summary statistics for the performance over episodes, focusing on either the training or testing dataset.

        @param dataset: Specifies whether the summary is for 'train' or 'test' data. Defaults to 'train'.
        @param percentage: The percentage of episodes at the beginning and end to compare. Defaults to 0.03 (3%).

        Prints the average and standard deviation of rewards and episode lengths, both overall and for the specified
        percentages at the start and end of training or testing.
        """
        episode_rewards = [np.mean([self.rewards[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]
        episode_lengths_list = [np.mean([self.episodes_length[env][i] for env in range(self.n_env)]) for i in range(self.n_episodes)]

        if dataset == "train":
            first_timesteps = int(self.n_episodes * percentage)
            last_timesteps = int(self.n_episodes * (1-percentage))

            first_rewards_sum = [np.sum(rewards) for rewards in episode_rewards[:first_timesteps]]
            last_rewards_sum = [np.sum(rewards) for rewards in episode_rewards[last_timesteps:]]

            first_avg_reward_per_episode = np.mean(first_rewards_sum)
            first_std_reward_per_episode = np.std(first_rewards_sum)
            first_avg_episode_length = np.mean(episode_lengths_list[:first_timesteps])
            first_std_episode_length = np.std(episode_lengths_list[:first_timesteps])

            last_avg_reward_per_episode = np.mean(last_rewards_sum)
            last_std_reward_per_episode = np.std(last_rewards_sum)
            last_avg_episode_length = np.mean(episode_lengths_list[last_timesteps:])
            last_std_episode_length = np.std(episode_lengths_list[last_timesteps:])

            print(f"- Total episodes: {self.n_episodes}")

            print(f"\nPerformance during first {str(percentage*100)}% of episodes")
            print(
                f"- Average reward per episode: {first_avg_reward_per_episode:.2f} \t\t Standard deviation: {first_std_reward_per_episode:.2f}")
            print(
                f"- Average episode length: {first_avg_episode_length:.2f} \t\t Standard deviation: {first_std_episode_length:.2f}")

            print(f"\nPerformance during last {str(percentage*100)}% of episodes")
            print(
                f"- Average reward per episode: {last_avg_reward_per_episode:.2f} \t\t Standard deviation: {last_std_reward_per_episode:.2f}")
            print(
                f"- Average episode length: {last_avg_episode_length:.2f} \t\t Standard deviation: {last_std_episode_length:.2f}")

        elif dataset == "test":
            rewards_sum = [np.sum(rewards) for rewards in episode_rewards]
            avg_reward_per_episode = np.mean(rewards_sum)
            std_reward_per_episode = np.std(rewards_sum)

            avg_episode_length = np.mean(episode_lengths_list)
            std_episode_length = np.std(episode_lengths_list)

            print(f"- Total episodes: {self.n_episodes}")
            print(
                f"- Average reward per episode: {avg_reward_per_episode:.2f} \t\t Standard deviation: {std_reward_per_episode:.2f}")
            print(
                f"- Average episode length: {avg_episode_length:.2f} \t\t Standard deviation: {std_episode_length:.2f}")
        else:
            print("Dataset parameter must be either 'train' or 'test'")


