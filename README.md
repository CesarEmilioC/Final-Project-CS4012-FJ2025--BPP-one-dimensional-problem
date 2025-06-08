# 🧠 RL-Based Hyper-Heuristic for the Bin Packing Problem (BPP)

This project implements a **Reinforcement Learning-based Hyper-Heuristic** for solving the **Bin Packing Problem (BPP)** using a set of classic heuristics and adapting the decision strategy through Q-learning.

The goal is to automatically learn **which heuristic to apply depending on the current state of the problem**, using features like item size distribution and bin usage.

---

## 📂 Project Structure

```
.
├── Code
|   ├── bpp.py
|   ├── hhproject.py             # Main script to run experiments
|   ├── phermes.py               # Base HyperHeuristic class
|   └── rl.py                    # RLHyperHeuristic with Q-learning logic
├── Instances/
│   └── BPP/
│       ├── Test set/
│       └── Training set/
├── Results/
|   └──Results (run number)
│     ├── q_values.json        # (Generated) Learned Q-values
│     └── resultados_globales.csv  # (Generated) Results per instance
├── licence.txt
├── requirements.txt
└── readme.txt               # Original readme (optional, replace with this .md)
```

---

## ⚙️ Requirements

- Python 3.7+
- `numpy`
- `pandas`

Install dependencies:

```bash
pip install requirements.txt
```

---

## 🧪 How to Run

Run the hyper-heuristic experiments with from the root directory:

```bash
python Code/hhproject.py
```

You can configure parameters like:
- Epsilon (exploration factor)
- Alpha (learning rate)
- Number of repetitions
- Training and test instance paths

All inside the `solveHH` function in `hhproject.py`.

---

## 🧠 Heuristics Supported

- **FFIT**: First Fit
- **BFIT**: Best Fit
- **WFIT**: Worst Fit
- **AWFIT**: Almost Worst Fit

---

## 🧬 Features Used for State Representation

Each state is represented using the following features from the BPP instance:

- `OPEN`: Ratio of open bins to total bins
- `LENGTH`: Normalized average item length
- `SMALL`: Ratio of items smaller than 50% of bin capacity
- `LARGE`: Ratio of items 50% or larger than bin capacity

These features allow the agent to learn context-aware decisions.

---

## 📊 Output Files

- `Results/Results (run number)/resultados_globales.csv`: Contains results per test instance:
  - Instance name
  - Number of items
  - Bins used
  - Heuristics used
  - Objective (waste) value

- `Results/Results (run number)/q_values.json`: Stores the final learned Q-values for each state-action pair.

