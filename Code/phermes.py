from __future__ import annotations
from typing import List
from copy import deepcopy
import numpy

# ====================================
# FILE: phermes.py
# DESCRIPTION:
# This file defines two base classes: `HyperHeuristic` and `Problem`.
# These classes provide generic interfaces for implementing
# hyper-heuristics and problems that can be solved by them.
# ====================================

class HyperHeuristic:
    """
    Base class for defining hyper-heuristics.

    A hyper-heuristic is a high-level algorithm that dynamically selects
    among different low-level heuristics based on the current problem state.
    """

    def __init__(self, features: List[str], heuristics: List[str]):
        """
        Initializes a new instance of the HyperHeuristic class.

        Parameters:
        - features (List[str]): List of feature names that describe the problem state.
        - heuristics (List[str]): List of available heuristics to apply to the problem.
        """
        self._features = deepcopy(features)
        self._heuristics = deepcopy(heuristics)

    def getFeatures(self) -> List[str]:
        """
        Returns a copy of the features used by this hyper-heuristic.

        Returns:
        - List[str]: List of feature names.
        """
        return deepcopy(self._features)

    def getHeuristics(self) -> List[str]:
        """
        Returns a copy of the available heuristics.

        Returns:
        - List[str]: List of heuristic names.
        """
        return deepcopy(self._heuristics)

    def getHeuristic(self, problem: Problem) -> str:
        """
        Abstract method to be implemented by subclasses.

        Given a problem, returns the name of the heuristic that should be applied
        in the current state of the problem.

        Parameters:
        - problem (Problem): The current problem instance.

        Returns:
        - str: Name of the selected heuristic.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")

# ====================================

class Problem:
    """
    Base class for defining problems that can be solved using heuristics
    or hyper-heuristics.
    """

    def solve(self, heuristic: str) -> None:
        """
        Applies a specific heuristic to modify or solve the problem state.

        Parameters:
        - heuristic (str): Name of the heuristic to apply.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")

    def solveHHA(self, hyperHeuristic: HyperHeuristic) -> None:
        """
        Solves the problem using a Type A hyper-heuristic.

        Type A: Heuristic selection based solely on features
        of the current problem state.

        Parameters:
        - hyperHeuristic (HyperHeuristic): An instance of a hyper-heuristic.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")

    def solveHHB(self, hyperHeuristic: HyperHeuristic) -> None:
        """
        Solves the problem using a Type B hyper-heuristic.

        Type B: Heuristic selection based on direct evaluation
        of objective value improvement after applying heuristics.

        Parameters:
        - hyperHeuristic (HyperHeuristic): An instance of a hyper-heuristic.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")

    def getFeature(self, feature: str) -> float:
        """
        Returns the current value of a specific problem feature.

        Parameters:
        - feature (str): Name of the feature.

        Returns:
        - float: Current value of the feature.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")

    def getObjValue(self) -> float:
        """
        Returns the current value of the problem's objective function.

        This value can represent the performance or quality of the
        current solution (e.g., wasted space, cost, time, etc.).

        Returns:
        - float: Objective function value.

        Raises:
        - Exception: If the method is not implemented.
        """
        raise Exception("Method not implemented yet.")