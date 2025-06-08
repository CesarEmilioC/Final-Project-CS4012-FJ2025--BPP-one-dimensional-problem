from phermes import Problem
from phermes import HyperHeuristic
import sys

# ====================================
class Item:
    """
    Represents an item with a unique ID and a given length.
    Used in the one-dimensional bin packing problem.
    """

    def __init__(self, id, length: int):
        self.id = id
        self.length = length

    def getId(self):
        """Returns the ID of the item."""
        return self.id

    def getLength(self):
        """Returns the length of the item."""
        return self.length

    def __str__(self):
        """String representation of the item."""
        return f"({self.id}, {self.length})"

# ====================================
class Bin:
    """
    Represents a bin with fixed capacity.
    Supports packing items and tracking remaining space.
    """

    def __init__(self, capacity: int):
        """
        Creates a new bin with a given capacity.
        """
        self._capacity = capacity
        self._items = []

    def getCapacity(self) -> int:
        """Returns the remaining capacity in the bin."""
        return self._capacity

    def canPack(self, item: Item) -> bool:
        """Checks if the item can fit in the bin."""
        return item.getLength() <= self._capacity

    def pack(self, item: Item) -> bool:
        """
        Packs an item into the bin if there's enough space.
        Returns True if successful, False otherwise.
        """
        if item.getLength() <= self._capacity:
            self._items.append(item)
            self._capacity -= item.getLength()
            return True
        return False

    def __str__(self):
        """String representation of the bin with its items."""
        text = "("
        for item in self._items:
            text += str(item)
        text += ")"
        return text

# ====================================
class BPP(Problem):
    """
    Represents a one-dimensional Bin Packing Problem (BPP).
    Includes methods to load, solve, and analyze the problem using heuristics.
    """

    def __init__(self, fileName: str):
        """
        Loads a BPP instance from a file.
        File format:
            Line 1: number of items
            Line 2: bin capacity
            Next lines: lengths of items
        """
        f = open(fileName, "r")
        lines = f.readlines()
        nbItems = int(lines[0].strip())
        self._capacity = int(lines[1].strip())
        self._items = [None] * nbItems
        for i in range(0, nbItems):
            size = int(lines[i + 2].strip())
            self._items[i] = Item(i, size)
        self._openBins = []
        self._closedBins = []

    def solve(self, heuristic: str) -> None:
        """
        Solves the BPP using the specified heuristic:
            - FFIT: First Fit
            - BFIT: Best Fit
            - WFIT: Worst Fit
            - AWFIT: Almost Worst Fit
        """
        while self._items:
            item = self._items.pop(0)
            bin = self._selectBin(item, heuristic)
            if bin is None:
                bin = Bin(self._capacity)
                self._openBins.append(bin)
            bin.pack(item)
            if bin.getCapacity() == 0:
                self._openBins.remove(bin)
                self._closedBins.append(bin)

    def solveHH(self, hyperHeuristic: HyperHeuristic) -> None:
        """
        Solves the BPP using a dynamic hyper-heuristic selector.
        The hyper-heuristic decides which heuristic to use at each step.
        """
        while self._items:
            item = self._items.pop(0)
            heuristic = hyperHeuristic.getHeuristic(self)
            bin = self._selectBin(item, heuristic)
            if bin is None:
                bin = Bin(self._capacity)
                self._openBins.append(bin)
            bin.pack(item)
            if bin.getCapacity() == 0:
                self._openBins.remove(bin)
                self._closedBins.append(bin)

    def getObjValue(self) -> float:
        """
        Returns the normalized waste score.
        Waste is defined as unused space, squared to penalize larger gaps.
        """
        waste = 0
        total_bins = len(self._openBins) + len(self._closedBins)
        if total_bins == 0:
            return 0.0
        for bin in self._openBins + self._closedBins:
            waste += ((self._capacity - bin.getCapacity()) / self._capacity) ** 2
        return waste / total_bins

    def getFeature(self, feature: str) -> float:
        """
        Extracts a feature from the problem state.
        Supported features:
            - "OPEN": Ratio of open bins to total bins
            - "LENGTH": Normalized average item length
            - "SMALL": Ratio of items < 50% of capacity
            - "LARGE": Ratio of items ≥ 50% of capacity
        """
        if feature == "OPEN":
            total = len(self._openBins) + len(self._closedBins)
            return len(self._openBins) / total if total > 0 else 0

        elif feature == "LENGTH":
            values = [item.getLength() for item in self._items]
            if values:
                return (sum(values) / len(values)) / max(values)
            return 0

        elif feature == "SMALL":
            count = sum(1 for item in self._items if item.getLength() < 0.5 * self._capacity)
            return count / len(self._items) if self._items else 0

        elif feature == "LARGE":
            count = sum(1 for item in self._items if item.getLength() >= 0.5 * self._capacity)
            return count / len(self._items) if self._items else 0

        else:
            raise Exception("Feature '" + feature + "' is not recognized by the system.")

    def _selectBin(self, item: Item, heuristic: str) -> Bin:
        """
        Selects a bin using the given heuristic.
        Returns a suitable open bin, or None if no bin can accommodate the item.
        """
        selected = None

        if heuristic == "FFIT":
            # First Fit: Selects the first open bin that fits
            for bin in self._openBins:
                if bin.canPack(item):
                    return bin

        elif heuristic == "BFIT":
            # Best Fit: Selects the bin with the least remaining space after packing
            waste = sys.maxsize
            for bin in self._openBins:
                if bin.canPack(item):
                    tmp = bin.getCapacity() - item.getLength()
                    if tmp < waste:
                        selected = bin
                        waste = tmp
            return selected

        elif heuristic == "WFIT":
            # Worst Fit: Selects the bin with the most remaining space after packing
            waste = -sys.maxsize - 1
            for bin in self._openBins:
                if bin.canPack(item):
                    tmp = bin.getCapacity() - item.getLength()
                    if tmp > waste:
                        selected = bin
                        waste = tmp
            return selected

        elif heuristic == "AWFIT":
            # Almost Worst Fit: Select the bin with the second largest remaining space
            candidates = [bin for bin in self._openBins if bin.canPack(item)]
            if not candidates:
                return None
            # Ordenar bins por espacio sobrante (capacidad - longitud item), descendente
            candidates.sort(key=lambda b: b.getCapacity() - item.getLength(), reverse=True)
            # Retornar segundo peor si hay al menos dos, sino el peor (primero)
            return candidates[1] if len(candidates) > 1 else candidates[0]

        else:
            raise Exception("Heuristic '" + heuristic + "' is not recognized by the system.")

    def __str__(self):
        """
        String representation of the current list of unpacked items.
        """
        return "(" + "".join(str(item) for item in self._items) + ")"