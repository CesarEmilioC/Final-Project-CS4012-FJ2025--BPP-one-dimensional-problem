# Import the Bin Packing Problem class
from bpp import BPP

# Module for file system interaction
import os

# Import the RL-based hyper-heuristic
from rl import RLHyperHeuristic

# Used to randomly shuffle instance files
import random

# Used to build and save tabular results
import pandas as pd


def solveHH(
    domain: str,
    folders: list,
    hyperHeuristic,
    repetitions: int = 1,
    reset_counters: bool = False,
    csv_path: str = "resultados_globales.csv",
    q_values_path: str = "q_values.json"
):
    """
    General function to solve problem instances using a hyper-heuristic.

    Args:
        domain (str): The problem domain, e.g., "BPP" (Bin Packing Problem).
        folders (list): List of folders containing .bpp instance files.
        hyperHeuristic: An instance of a hyper-heuristic (standard or RL-based).
        repetitions (int): Number of times to solve each instance (for averaging or robustness).
        reset_counters (bool): Whether to reset heuristic usage counters before each run.
        csv_path (str): Output path for saving the CSV file with summary results.
        q_values_path (str): Output path for saving learned Q-values (for RL-based HH).
    """
    results = []  # Stores summary statistics for each run

    for folder in folders:
        # List all .bpp files in the current folder
        files = [f for f in os.listdir(folder) if f.endswith(".bpp")]
        random.shuffle(files)  # Shuffle instance order for variability

        for idx, file in enumerate(files):
            for i in range(repetitions):

                # Optionally reset internal counters in the hyper-heuristic
                if reset_counters:
                    hyperHeuristic.reset_counters()

                # Build the full path to the instance file
                instance_path = os.path.join(folder, file)

                # Load and solve the instance
                if domain == "BPP":
                    problem = BPP(instance_path)
                else:
                    raise Exception(f"Unsupported domain: {domain}")

                # Apply the hyper-heuristic to solve the problem
                problem.solveHH(hyperHeuristic)

                # Extract performance metrics after solving
                obj = problem.getObjValue()  # Objective value (waste)
                total_bins = len(problem._openBins) + len(problem._closedBins)  # Total bins used
                total_items = getattr(hyperHeuristic, 'steps', 0)  # Number of heuristic actions applied
                heuristics_used = hyperHeuristic.get_heuristic_counts().copy()  # Usage count for each heuristic

                # Store the run result
                results.append({
                    "INSTANCE": file,
                    "ITEMS": total_items,
                    "BINS": total_bins,
                    "HEURISTICS": heuristics_used,
                    "WASTE (OBJ)": round(obj, 4)
                })

    # Convert results to a DataFrame and print as a Markdown table
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save summary to CSV
    df.to_csv(csv_path, index=False)

    # Save Q-values (only meaningful for RL-based hyper-heuristics)
    hyperHeuristic.save_q_values(q_values_path)

    return df  # Return the summary DataFrame for further use


# This block runs only if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # List of heuristic strategies available
    heuristics = ["FFIT", "BFIT", "WFIT", "AWFIT"]

    # List of folders containing instance files for training and testing
    folders = ["Instances/Bpp/Training set", "Instances/Bpp/Test set"]

    # Instantiate an RL-based hyper-heuristic with a small epsilon (for exploration)
    rl_hh = RLHyperHeuristic(heuristics, epsilon=0.1)

    # Run the solver on all folders and save results
    solveHH(
        domain="BPP",
        folders=folders,
        hyperHeuristic=rl_hh,
        repetitions=3,  # Solve each instance 3 times
        reset_counters=True,  # Reset usage counters between runs
        csv_path="Results/resultados_globales.csv",  # Output results file
        q_values_path="Results/q_values.json"  # Output Q-values file
    )
