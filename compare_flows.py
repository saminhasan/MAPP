# File: compare_flows.py

import time
import numpy as np

# Import the flow solvers from your two modules
from nx_flow import solve_nx_flow
from sparse_flow import solve_sparse_flow

# Suppose we also import the random point generator from a shared module
from nx_flow import generate_random_points


def compare_paths(paths_a, paths_b):
    """
    Compare two sets of paths for exact equality.
    If path order or path shape can differ, you'll need a more robust approach.
    """
    if len(paths_a) != len(paths_b):
        return False, "Different number of paths"

    for i in range(len(paths_a)):
        if len(paths_a[i]) != len(paths_b[i]):
            return False, f"Paths {i} have different lengths"
        if not np.array_equal(paths_a[i], paths_b[i]):
            return False, f"Paths {i} differ in coordinates"

    return True, "All paths match exactly!"


def main():
    # 1) Pick a scenario
    dim = 3
    size = 25
    n_agents = 10
    seed = 3

    # 2) Generate one set of starts/goals
    starts, goals, _ = generate_random_points(size, dim, n_agents, seed=seed)

    # 3) Solve with Nx flow
    nx_paths, nx_time = solve_nx_flow(starts, goals, dim=dim, size=size)
    print(f"[nx_flow]   total time = {round(nx_time, 3)}s, found {len(nx_paths)} paths")

    # 4) Solve with SciPy flow
    sp_paths, sp_time = solve_sparse_flow(starts, goals, dim=dim, size=size)
    print(f"[sparse_flow] total time = {round(sp_time, 3)}s, found {len(sp_paths)} paths")

    # 5) Compare times
    if nx_time < sp_time:
        print(f"NetworkX flow was faster by {round(sp_time - nx_time, 3)}s.")
    elif sp_time < nx_time:
        print(f"SciPy flow was faster by {round(nx_time - sp_time, 3)}s.")
    else:
        print("Both took exactly the same time. (Unlikely, but hey!)")

    # 6) Compare paths
    same, msg = compare_paths(nx_paths, sp_paths)
    if same:
        print("SUCCESS: Both methods returned the same paths!")
    else:
        print(f"WARNING: {msg}")

    for s, n in zip(sp_paths, nx_paths):
        print(s[0], n[0], s[-1], n[-1])


if __name__ == "__main__":
    main()
