"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from logs import CustomFormatter

############################### LOGGER

logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

logger = logging.getLogger("MAB Application")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

#--------------------------------------#

class Bandit(ABC):
    """
    Abstract class representing a multi-armed bandit.
    """
    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit with the probability distribution of rewards.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the bandit.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Pull an arm of the bandit and return the index of the chosen arm.
        """
        pass

    @abstractmethod
    def update(self, bandit_idx, reward):
        """
        Update the bandit's internal state based on the reward received from pulling a specific arm.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials):
        """
        Run the bandit experiment for a specified number of trials.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Report the results of the bandit experiment.
        """
        pass

    @abstractmethod
    def cumulative_regret(self, rewards):
        """
        Calculate the cumulative regret based on the rewards obtained.
        """
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy multi-armed bandit algorithm.
    """
    def __init__(self, p, epsilon):
        """
        Initialize the Epsilon-Greedy bandit with the probability distribution and epsilon value.
        """
        if len(p) < 2:
            raise ValueError("Number of bandit arms must be at least 2")
        super().__init__(p)
        self.epsilon = epsilon
        self.mean = np.zeros(len(p))  # Initialize mean rewards for each bandit arm
        self.N = np.zeros(len(p))      # Initialize counts for each bandit arm
        self.p = p  # Store the probability distribution of rewards for each bandit arm

    def __repr__(self):
        """
        Return a string representation of the Epsilon-Greedy bandit.
        """
        return f"Epsilon Greedy with epsilon={self.epsilon}"

    def pull(self):
        """
        Choose whether to explore or exploit based on epsilon value and return the chosen arm.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.p))
        else:
            return np.argmax(self.mean)

    def update(self, bandit_idx, reward):
        """
        Update the bandit's mean reward estimate based on the reward received.
        """
        self.N[bandit_idx] += 1
        self.mean[bandit_idx] = (1 - 1.0/self.N[bandit_idx]) * self.mean[bandit_idx] + 1.0/self.N[bandit_idx] * reward

    def experiment(self, num_trials):
        """
        Run the Epsilon-Greedy bandit experiment for a specified number of trials.
        """
        rewards = np.zeros(num_trials)
        chosen_bandits = np.zeros(num_trials)
        cumulative_rewards = np.zeros(num_trials)

        for i in range(num_trials):
            bandit_idx = self.pull()
            reward = np.random.normal(self.p[bandit_idx], 1)
            self.update(bandit_idx, reward)

            rewards[i] = reward
            chosen_bandits[i] = bandit_idx
            cumulative_rewards[i] = np.sum(rewards)

        return rewards, chosen_bandits, cumulative_rewards

    def report(self, rewards):
        """
        Report the results of the Epsilon-Greedy bandit experiment.
        """
        logger.info(f"Epsilon Greedy: Cumulative Reward = {np.sum(rewards)}")
        regret = self.cumulative_regret(rewards)
        logger.info(f"Epsilon Greedy: Cumulative Regret = {np.sum(regret)}")

    def cumulative_regret(self, rewards):
        """
        Calculate the cumulative regret for the Epsilon-Greedy bandit experiment.
        """
        return np.max(self.p) * np.arange(1, len(rewards)+1) - np.cumsum(rewards)

    def plot_learning_process(self, rewards):
        """
        Visualize the learning process of the Epsilon-Greedy bandit.
        """
        plt.plot(np.cumsum(rewards))
        plt.title("Epsilon Greedy Learning Process")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling multi-armed bandit algorithm.
    """
    def __init__(self, p, precision):
        """
        Initialize the Thompson Sampling bandit with the probability distribution and precision value.
        """
        super().__init__(p)
        self.precision = precision
        self.mean = np.zeros(len(p))  # Initialize mean rewards for each bandit arm
        self.N = np.zeros(len(p))      # Initialize counts for each bandit arm
        self.p = p  # Store the probability distribution of rewards for each bandit arm

    def __repr__(self):
        """
        Return a string representation of the Thompson Sampling bandit.
        """
        return f"Thompson Sampling with precision={self.precision}"

    def pull(self):
        """
        Sample from the posterior distribution and return the arm with the highest sampled value.
        """
        samples = np.random.normal(self.mean, self.precision)
        return np.argmax(samples)

    def update(self, bandit_idx, reward):
        """
        Update the bandit's mean reward estimate based on the reward received.
        """
        self.N[bandit_idx] += 1
        self.mean[bandit_idx] = (1 - 1.0/self.N[bandit_idx]) * self.mean[bandit_idx] + 1.0/self.N[bandit_idx] * reward

    def experiment(self, num_trials):
        """
        Run the Thompson Sampling bandit experiment for a specified number of trials.
        """
        rewards = np.zeros(num_trials)
        chosen_bandits = np.zeros(num_trials)
        cumulative_rewards = np.zeros(num_trials)

        for i in range(num_trials):
            bandit_idx = self.pull()
            reward = np.random.normal(self.p[bandit_idx], 1)
            self.update(bandit_idx, reward)

            rewards[i] = reward
            chosen_bandits[i] = bandit_idx
            cumulative_rewards[i] = np.sum(rewards)

        return rewards, chosen_bandits, cumulative_rewards

    def report(self, rewards):
        """
        Report the results of the Thompson Sampling bandit experiment.
        """
        logger.info(f"Thompson Sampling: Cumulative Reward = {np.sum(rewards)}")
        regret = self.cumulative_regret(rewards)
        logger.info(f"Thompson Sampling: Cumulative Regret = {np.sum(regret)}")

    def cumulative_regret(self, rewards):
        """
        Calculate the cumulative regret for the Thompson Sampling bandit experiment.
        """
        return np.max(self.p) * np.arange(1, len(rewards)+1) - np.cumsum(rewards)

    def plot_learning_process(self, rewards):
        """
        Visualize the learning process of the Thompson Sampling bandit.
        """
        plt.plot(np.cumsum(rewards))
        plt.title("Thompson Sampling Learning Process")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

#--------------------------------------#

def main():
    Bandit_Reward = [1, 2, 3, 4]
    num_trials = 20000
    epsilon = 0.1
    precision = 0.1

    egreedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    ts_bandit = ThompsonSampling(Bandit_Reward, precision)

    egreedy_rewards, _, egreedy_cumulative_rewards = egreedy_bandit.experiment(num_trials)
    ts_rewards, _, ts_cumulative_rewards = ts_bandit.experiment(num_trials)

    egreedy_bandit.plot_learning_process(egreedy_rewards)
    ts_bandit.plot_learning_process(ts_rewards)

    egreedy_bandit.report(egreedy_rewards)
    ts_bandit.report(ts_rewards)

    # Store rewards in CSV
    df = pd.DataFrame({
        "Bandit": ["Bandit"] * num_trials * 2,
        "Reward": np.concatenate((egreedy_rewards, ts_rewards)),
        "Algorithm": ["Epsilon Greedy"] * num_trials + ["Thompson Sampling"] * num_trials
    })
    df.to_csv("rewards.csv", index=False)

    # Log cumulative reward and regret
    logger.info(f"Epsilon Greedy - Cumulative Reward: {np.sum(egreedy_rewards)}")
    logger.info(f"Thompson Sampling - Cumulative Reward: {np.sum(ts_rewards)}")
    egreedy_regret = egreedy_bandit.cumulative_regret(egreedy_rewards)
    ts_regret = ts_bandit.cumulative_regret(ts_rewards)
    logger.info(f"Epsilon Greedy - Cumulative Regret: {np.sum(egreedy_regret)}")
    logger.info(f"Thompson Sampling - Cumulative Regret: {np.sum(ts_regret)}")

    # Plot cumulative rewards comparison
    plt.plot(np.cumsum(egreedy_rewards), label='E-Greedy')
    plt.plot(np.cumsum(ts_rewards), label='Thompson Sampling')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()


##==== better implementation plan ====##
"""
Break down the code into distinct modules to enhance clarity and maintainability.
Each module should serve a specific purpose, such as implementing bandit algorithms,
configuring experiments, visualizing results, and reporting outcomes.

Implement an experiment configuration mechanism using JSON or YAML files to define experiment
parameters such as bandit arm probabilities, algorithm selections, epsilon values, precision levels,
and trial counts. This setup enables seamless experimentation without the need for code alterations.
"""