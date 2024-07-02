import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class ExperienceDict:
    """
    Class that stores and manages experiences from multiple environments.

    @param n_envs: Number of environments.
    @param batch_size: Total number of trajectories to consider the batch full.
    @param minibatch_size: Number of trajectories in each minibatch for training.
    """
    def __init__(self, n_envs, batch_size, minibatch_size):
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.current_batch_size = 0

        self.experienceDict = {env: {key: [] for key in ["states", "actions", "action_probs", "rewards", "values", "dones"]} for env in range(self.n_envs)}

    def clear(self):
        """
        Clears the experience dictionary and resets the current batch size.
        """
        self.experienceDict = {env: {key: [] for key in ["states", "actions", "action_probs", "rewards", "values", "dones"]} for env in range(self.n_envs)}
        self.current_batch_size = 0

    def get(self, key, env=-1, return_type="np"):
        """
        Retrieve and concatenate data from the experience buffer based on specified key and environment.

        @param key: Type of data to retrieve (e.g., "states", "actions").
        @param env: Specific environment ID to fetch data from; if -1, fetches from all environments.
        @param return_type: Format to return the data in ("np" for numpy array, "torch" for torch tensor).
        @return: The requested data as a numpy array or a torch tensor.
        """
        if env == -1:
            concat_data = np.concatenate([self.experienceDict[i][key] for i in range(self.n_envs)])
        else:
            concat_data = np.array(self.experienceDict[env][key])
        if return_type == "np":
            return concat_data
        elif return_type == "torch":
            return torch.tensor(concat_data, dtype=torch.float32)

    def is_full(self):
        """
        Checks if the current batch size has reached or exceeded the set batch size.

        @return: Boolean indicating whether the batch size limit has been reached.
        """
        return self.current_batch_size >= self.batch_size

    def appendTrajectory(self, state, action, action_prob, reward, value, done):
        """
        Append a trajectory (experience at a timestep) for each environment.

        @param state: Array of states (observations) for each environment.
        @param action: Array of actions for each environment.
        @param action_prob: Array of action probabilities for each environment.
        @param reward: Array of rewards for each environment.
        @param value: Array of value estimates for each environment.
        @param done: Array of done signals (indicating end of episode) for each environment.
        """
        self.current_batch_size += self.n_envs

        for env in range(self.n_envs):
            self.experienceDict[env]["states"].append(state[env])
            self.experienceDict[env]["actions"].append(action[env])
            self.experienceDict[env]["action_probs"].append(action_prob[env])
            self.experienceDict[env]["rewards"].append(reward[env])
            self.experienceDict[env]["values"].append(value[env])
            self.experienceDict[env]["dones"].append(done[env])

    def compute_advantage_and_returns(self, next_values, gamma, lambda_):
        """
        Calculate the generalized advantage estimation (GAE) and returns for each environment.

        @param next_values: The value estimates for the next state (beyond the current batch of data).
        @param gamma: Discount factor for rewards.
        @param lambda_: Smoothing parameter for GAE.
        @return: Two torch tensors containing the advantages and returns respectively.
        """
        advantages = []
        returns = []

        # GAE computed separately for each environment
        for env_index in range(self.n_envs):
            rewards = self.get("rewards", env_index, return_type="np")
            values = self.get("values", env_index, return_type="np")
            dones = self.get("dones", env_index, return_type="np")

            env_advantages = np.zeros_like(rewards, dtype=np.float64)
            next_value = next_values[env_index]
            gae = 0

            # Calculate returns by iterating backwards through the rewards
            for i in reversed(range(len(rewards))):

                # if this is the last position of the batch retrieve the next value
                if i != len(rewards) - 1:
                    next_value = values[i + 1]

                # update next_value and gae according to the fact that this may be a terminal state
                next_value = next_value * (1 - dones[i])
                gae = gae * (1 - dones[i])

                # compute advantages
                delta = rewards[i] + gamma * next_value - values[i]
                env_advantages[i] = gae = delta + gamma * lambda_ * gae

            # normalize the advantage
            env_advantages = env_advantages - np.mean(env_advantages) / (np.std(env_advantages) + 1e-10)

            # concatenate the environment advantages and returns to the full lists
            advantages.extend(env_advantages)
            returns.extend(env_advantages+values)

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def get_batches(self, device, next_values, td_gamma=0.999, gae_lambda=0.95, shuffle=True):
        """
        Create batches of data for training using the stored experiences.

        @param device: The device (CPU or GPU) to transfer the tensors to.
        @param next_values: The value estimates for the next state.
        @param td_gamma: discount factor for rewards.
        @param gae_lambda: Smoothing parameter for GAE.
        @param shuffle: Whether to shuffle the batches or not.
        @return: A DataLoader containing the batches of experiences for training.
        """
        states = self.get("states", return_type="torch").to(device)
        actions = self.get("actions", return_type="torch").to(device)
        action_probs = self.get("action_probs", return_type="torch").to(device)
        advantages, returns = self.compute_advantage_and_returns(next_values, td_gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = returns.to(device)

        dataset = TensorDataset(states, actions, action_probs, advantages, returns)
        return DataLoader(dataset, batch_size=self.minibatch_size, shuffle=shuffle)