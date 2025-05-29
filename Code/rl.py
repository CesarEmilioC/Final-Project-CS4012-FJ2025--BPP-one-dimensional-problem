from phermes import HyperHeuristic
import random
import json
import numpy as np

class RLHyperHeuristic(HyperHeuristic):
    """
    Reinforcement Learning-based Hyper-Heuristic for the Bin Packing Problem.

    This class uses an epsilon-greedy strategy with softmax exploration to dynamically select 
    among low-level heuristics (like First Fit, Best Fit, etc.), updating its Q-values based 
    on performance improvements (reduction in waste).
    """

    def __init__(self, heuristics, epsilon=0.3, alpha=0.5, epsilon_decay=0.98, epsilon_min=0.05):
        """
        Initializes the RL hyper-heuristic with parameters for learning and exploration.

        Args:
            heuristics (list): List of heuristic names to choose from.
            epsilon (float): Initial probability of exploring (vs. exploiting).
            alpha (float): Learning rate for Q-value updates.
            epsilon_decay (float): Decay factor for epsilon after each step.
            epsilon_min (float): Minimum value for epsilon (exploration floor).
        """
        self.heuristics = heuristics
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_values = {h: 0.0 for h in heuristics}  # Q-values for each heuristic
        self.last_waste = None                       # Waste from previous step
        self.last_heuristic = None                   # Last heuristic used
        self.steps = 0                               # Total decision steps taken
        self.heuristic_counts = {h: 0 for h in heuristics}  # Frequency of heuristic usage

    def getHeuristic(self, problem):
        """
        Selects the next heuristic to apply based on the current state and Q-values.

        Args:
            problem: The BPP problem instance.

        Returns:
            str: The selected heuristic.
        """
        current_waste = problem.getObjValue()

        # Compute reward and update Q-value if previous step exists
        if self.last_waste is not None and self.last_heuristic is not None:
            reward = self.last_waste - current_waste
            if reward == 0:
                reward = -0.05  # Small penalty to discourage stagnation
            self._update_q_value(self.last_heuristic, reward)

        # Decide between exploration and exploitation
        if random.random() < self.epsilon:
            heuristic = self._softmax_select()
        else:
            heuristic = max(self.q_values, key=self.q_values.get)

        # Update internal state
        self.last_heuristic = heuristic
        self.last_waste = current_waste
        self.steps += 1
        self.heuristic_counts[heuristic] += 1

        # Apply epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return heuristic

    def _update_q_value(self, heuristic, reward):
        """
        Performs Q-learning update for the selected heuristic.

        Args:
            heuristic (str): The heuristic to update.
            reward (float): The observed reward.
        """
        current_q = self.q_values[heuristic]
        self.q_values[heuristic] = current_q + self.alpha * (reward - current_q)

    def get_heuristic_counts(self):
        """
        Returns a copy of how many times each heuristic has been used.

        Returns:
            dict: Heuristic usage count.
        """
        return self.heuristic_counts.copy()

    def save_q_values(self, filepath="q_values.json"):
        """
        Saves the current Q-values to a JSON file.

        Args:
            filepath (str): Path to the file where Q-values will be saved.
        """
        with open(filepath, "w") as f:
            json.dump(self.q_values, f, indent=4)

    def load_q_values(self, filepath="q_values.json"):
        """
        Loads Q-values from a JSON file if it exists.

        Args:
            filepath (str): Path to the file from which to load Q-values.
        """
        try:
            with open(filepath, "r") as f:
                self.q_values = json.load(f)
            # Ensure all keys are strings and values are floats
            self.q_values = {str(k): float(v) for k, v in self.q_values.items()}
        except FileNotFoundError:
            print(f"File {filepath} not found. Initial Q-values will be used.")

    def reset_counters(self):
        """
        Resets internal counters, heuristic tracking, and step count.
        Useful for restarting experiments.
        """
        self.steps = 0
        self.heuristic_counts = {h: 0 for h in self.heuristics}
        self.last_heuristic = None
        self.last_waste = None

    def _softmax_select(self):
        """
        Selects a heuristic using softmax probability distribution based on Q-values.

        Returns:
            str: Selected heuristic.
        """
        q_vals = np.array([self.q_values[h] for h in self.heuristics])
        exp_q = np.exp(q_vals - np.max(q_vals))  # Numerical stability
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.heuristics, p=probs)