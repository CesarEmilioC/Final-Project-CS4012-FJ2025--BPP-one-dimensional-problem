from phermes import HyperHeuristic
import random
import json
import numpy as np

class RLHyperHeuristic(HyperHeuristic):
    """
    Reinforcement Learning-based Hyper-Heuristic for the Bin Packing Problem.

    This version uses problem state features as input to a state-action Q-table for
    heuristic selection. It follows an epsilon-greedy policy with softmax exploration.
    """

    def __init__(self, heuristics, epsilon=0.5, alpha=0.5, epsilon_decay=0.98, epsilon_min=0.05):
        self.heuristics = heuristics
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_values = {}  # Nested dict: {state_str: {heuristic: q_value}}
        self.last_state = None
        self.last_heuristic = None
        self.last_waste = None

        self.steps = 0
        self.heuristic_counts = {h: 0 for h in heuristics}

    def getHeuristic(self, problem):
        """
        Selects the next heuristic based on the current problem state features and Q-values.
        """
        current_state = self._get_state_from_features(problem)
        current_waste = problem.getObjValue()

        # Update Q-value if previous step exists
        if self.last_state is not None and self.last_heuristic is not None:
            
            # Calculate raw improvement in waste
            raw_reward = self.last_waste - current_waste
            
            # Normalize the reward relative to previous waste to avoid scale issues
            if self.last_waste > 1e-5:
                normalized_reward = raw_reward / self.last_waste
            else:
                normalized_reward = 0.0
            
            # Clip reward to the range [-1, 1] to prevent extreme Q-value updates
            reward = max(min(normalized_reward, 1.0), -1.0)
            
            self._update_q_value(self.last_state, self.last_heuristic, reward)

        # Make sure state exists in Q-table
        if current_state not in self.q_values:
            self.q_values[current_state] = {h: 0.0 for h in self.heuristics}

        # Select heuristic
        if random.random() < self.epsilon:
            heuristic = random.choice(self.heuristics)
        else:
            heuristic = max(self.q_values[current_state], key=self.q_values[current_state].get)

        # Update internal state
        self.last_state = current_state
        self.last_heuristic = heuristic
        self.last_waste = current_waste
        self.steps += 1
        self.heuristic_counts[heuristic] += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return heuristic
    
    def _discretize(self, value, step=0.1):
        return round(round(value / step) * step, 2)

    def _get_state_from_features(self, problem):
        """
        Extracts and discretizes a state vector from problem features.

        Returns:
            str: A hashable string representing the discretized state.
        """

        features = [
            self._discretize(problem.getFeature("OPEN")),
            self._discretize(problem.getFeature("LENGTH")),
            self._discretize(problem.getFeature("SMALL")),
            self._discretize(problem.getFeature("LARGE"))
        ]
        
        return str(features)  # e.g., "[0.3, 0.7, 0.5, 0.5]"

    def _update_q_value(self, state, heuristic, reward):
        """
        Updates Q-value for a given state-action pair.

        Args:
            state (str): Discretized state string.
            heuristic (str): Chosen heuristic.
            reward (float): Observed reward.
        """
        current_q = self.q_values[state][heuristic]
        self.q_values[state][heuristic] = current_q + self.alpha * (reward - current_q)

    def get_heuristic_counts(self):
        return self.heuristic_counts.copy()

    def save_q_values(self, filepath="q_values.json"):
        with open(filepath, "w") as f:
            json.dump(self.q_values, f, indent=4)

    def reset_counters(self):
        self.steps = 0
        self.heuristic_counts = {h: 0.0 for h in self.heuristics}
        self.last_heuristic = None
        self.last_waste = None
        self.last_state = None