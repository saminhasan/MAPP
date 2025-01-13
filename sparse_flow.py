import time
import random
import numpy as np
from itertools import product

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_flow

import matplotlib.pyplot as plt
from matplotlib import colors


class PathNotFound(Exception):
    pass


def timer(start, label=""):
    """
    Prints the elapsed time since 'start' with an optional label.
    Returns the current time so you can measure again.
    """
    now = time.perf_counter()
    print(f"{label} : {round(now - start, 3)}s.")
    return now


def generate_random_points(size, dim, n_agents, seed=None):
    """
    Generates:
        - 'inside_nodes': The set of points strictly inside the grid
        - 'edge_nodes': The set of points on the boundary
        - 'starts': Randomly-chosen start locations (from 'inside_nodes')
        - 'goals': Randomly-chosen goal locations (from 'edge_nodes')
    """
    if seed is not None:
        random.seed(seed)

    # All points in an n-dim grid of 'size' in each dimension
    nodes = list(product(*[range(size)] * dim))

    # A smaller bounding box for 'inside' to ensure there's an actual 'edge'
    inside_nodes = list(product(*[range(2, size - 2)] * dim))
    edge_nodes = set(nodes) - set(inside_nodes)  # all points not inside

    starts = [random.choice(inside_nodes) for _ in range(n_agents)]
    goals = [random.choice(list(edge_nodes)) for _ in range(n_agents)]

    return starts, goals, nodes


def coord_to_index(coord, size, dim):
    """
    Converts an n-D coordinate (tuple) to a single integer index
    (row-major ordering).
    """
    # Example (2D): (x, y) -> x * size + y
    # Example (3D): (x, y, z) -> x * size^2 + y * size + z
    # etc.
    idx = 0
    mul = 1
    for c in reversed(coord):
        idx += c * mul
        mul *= size
    return idx


def index_to_coord(index, size, dim):
    """
    Inverse of coord_to_index (row-major).
    """
    coords = []
    for _ in range(dim):
        coords.append(index % size)
        index //= size
    return tuple(reversed(coords))


def build_grid_matrix(size, dim, starts, goals):
    """
    Builds a directed grid graph in an n-D, size^dim domain.
    Each edge has capacity = 1.

    Also adds a super-source and super-sink:
      - super_source_index = total_nodes
      - super_sink_index   = total_nodes + 1

    Returns:
      A (CSR matrix)        : adjacency matrix with capacities
      node_count (int)      : total_nodes (without super-source/sink)
      super_source_index    : index of super-source in adjacency
      super_sink_index      : index of super-sink in adjacency
      all_nodes (list)      : list of coordinates (tuples) in the grid
    """
    # total grid nodes
    node_count = size**dim

    # We'll collect row, col, data for edges in lists
    row = []
    col = []
    data = []

    # For each coordinate in the grid, link to neighbors in each dimension
    all_nodes = [index_to_coord(i, size, dim) for i in range(node_count)]

    for i, coord in enumerate(all_nodes):
        # For each dimension, try +1 and -1 neighbor to form edges
        for d in range(dim):
            neighbor_plus = list(coord)
            neighbor_plus[d] += 1
            if neighbor_plus[d] < size:
                j = coord_to_index(tuple(neighbor_plus), size, dim)
                # add edge i->j (capacity=1)
                row.append(i)
                col.append(j)
                data.append(1)

            neighbor_minus = list(coord)
            neighbor_minus[d] -= 1
            if neighbor_minus[d] >= 0:
                j = coord_to_index(tuple(neighbor_minus), size, dim)
                # add edge i->j (capacity=1)
                row.append(i)
                col.append(j)
                data.append(1)

    # Add super-source, super-sink
    super_source_index = node_count
    super_sink_index = node_count + 1

    # For each start, add edge (super_source -> start)
    for s in starts:
        s_idx = coord_to_index(s, size, dim)
        row.append(super_source_index)
        col.append(s_idx)
        data.append(1)

    # For each goal, add edge (goal -> super_sink)
    for g in goals:
        g_idx = coord_to_index(g, size, dim)
        row.append(g_idx)
        col.append(super_sink_index)
        data.append(1)

    total_nodes_with_ss = node_count + 2

    # Build a COO matrix then convert to CSR
    A_coo = coo_matrix((data, (row, col)), shape=(total_nodes_with_ss, total_nodes_with_ss), dtype=np.int32)
    A = A_coo.tocsr()  # Convert to CSR for maximum_flow

    return A, node_count, super_source_index, super_sink_index, all_nodes


def extract_paths_from_flow_matrix(flow_matrix, size, dim, starts, goals, super_source_idx, super_sink_idx):
    """
    Reconstructs paths by following edges with flow > 0
    directly from the flow_matrix returned by SciPy maximum_flow.

    flow_matrix        : 2D NumPy array from `res.flow`
    size, dim          : grid parameters (for indexing)
    starts, goals      : lists of coordinate tuples
    super_source_idx   : integer index of super-source
    super_sink_idx     : integer index of super-sink

    Returns a list of np.ndarray (one path array per agent)
    """
    # Build an index <-> coord map for the "real" grid nodes
    # (excluding super-source, super-sink)
    node_count = size**dim

    def idx_to_coord(idx):
        return index_to_coord(idx, size, dim)

    # Put goals in a set for quick membership tests
    goals_set = set(goals)

    paths = []

    for start in starts:
        path = []
        current_coord = start
        current_idx = coord_to_index(current_coord, size, dim)

        while True:
            path.append(current_coord)
            if current_coord in goals_set:
                # Reached a goal
                break

            # Find an outgoing edge with flow>0 from current_idx
            # search only among 'real' nodes and super_sink
            row_flow = flow_matrix.getrow(current_idx).toarray()[0]
            next_idx = None
            # We can scan row_flow for a positive flow
            # ignoring super_source because we never want to go back

            for j in np.where(row_flow > 0)[0]:
                if j != super_source_idx:  # skip going back to super-source
                    next_idx = j
                    break

            if next_idx is None:
                # Path reconstruction failed
                path = []
                break

            # Convert next_idx -> coordinate (unless it's super-sink)
            if next_idx == super_sink_idx:
                # The next node is super-sink => we reached the end
                # (we only consider this valid if we're actually a goal)
                # But the check above was "if current_coord in goals_set"...
                # so typically we wouldn't break here unless we want to handle
                # corner cases (some flows might go directly into sink).
                break

            current_coord = idx_to_coord(next_idx)
            current_idx = next_idx

        if not path or path[-1] not in goals_set:
            raise PathNotFound(f"No valid path found from start={start}")

        paths.append(np.array(path))

    return paths


def plot_paths(paths, starts, goals, dim, n_agents, title="Multi-Agent Paths"):
    """
    Plots the paths in either 2D or 3D with a single legend showing
    'Start', 'Path', 'Goal'. Each agent gets a unique color.
    """
    fig = plt.figure(title)

    # Create axes for 2D or 3D
    if dim == 2:
        axes = fig.add_subplot()
        axes.set_xlabel("X Axis")
        axes.set_ylabel("Y Axis")
    elif dim == 3:
        axes = fig.add_subplot(projection="3d")
        axes.set_xlabel("X Axis")
        axes.set_ylabel("Y Axis")
        axes.set_zlabel("Z Axis")
    else:
        raise ValueError("This plotting function only supports 2D or 3D.")

    axes.set_title(f"{dim}D plot of Paths")

    # Color normalization (range 0..n_agents)
    cn = colors.Normalize(0, n_agents)

    # For legend
    handles, labels = [], []

    # Plot each agent’s path
    for i, path in enumerate(paths):
        color = plt.cm.hsv(cn(i))  # Pick a color for the agent's path

        if dim == 2:
            # Scatter start
            start_obj = axes.scatter(path[0][0], path[0][1], color=color, marker="x", s=200, label="Start")
            # Path line
            path_objs = axes.plot(path[:, 0], path[:, 1], color=color, lw=2, label="Path")
            # Scatter goal
            goal_obj = axes.scatter(path[-1][0], path[-1][1], color=color, marker="*", s=200, label="Goal")
        else:  # dim == 3
            start_obj = axes.scatter(path[0][0], path[0][1], path[0][2], color=color, marker="x", s=200, label="Start")
            path_objs = axes.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2, label="Path")
            goal_obj = axes.scatter(path[-1][0], path[-1][1], path[-1][2], color=color, marker="*", s=200, label="Goal")

        # Add "Start", "Path", "Goal" to legend only once
        if i == 0:
            # path_objs is a list from plt.plot, so extract the line handle
            for obj in [start_obj, path_objs[0], goal_obj]:
                handles.append(obj)
                labels.append(obj.get_label())

    # Remove duplicate labels, if any
    by_label = dict(zip(labels, handles))
    axes.legend(by_label.values(), by_label.keys(), loc="best")

    plt.show()


def check_paths(starts, goals, paths):
    """
    Checks that every path starts with a valid 'start' and
    ends with a valid 'goal', removing them as it goes.
    If at the end, no starts/goals remain, we know each
    was used exactly once. Otherwise, an error is raised.
    """
    # Convert to sets for O(1) membership checks and removals
    remaining_starts = set(starts)
    remaining_goals = set(goals)

    # Print lengths
    print("Length of starts:", len(starts))
    print("Length of goals:", len(goals))
    print("Length of remaining_starts:", len(remaining_starts))
    print("Length of remaining_goals:", len(remaining_goals))
    if len(remaining_starts) != len(starts):
        print("duplicate starts")
    if len(remaining_goals) != len(goals):
        print("duplicate goals")
    for i, path in enumerate(paths):
        # path is a NumPy array of coordinates
        start_node = tuple(path[0])
        goal_node = tuple(path[-1])

        # Remove the start from remaining_starts
        if start_node in remaining_starts:
            remaining_starts.remove(start_node)
        else:
            raise ValueError(f"Path {i} start {start_node} is not in 'starts' or was used already!")

        # Remove the goal from remaining_goals
        if goal_node in remaining_goals:
            remaining_goals.remove(goal_node)
        else:
            print(goals, starts)
            print(f"Path {i} goal {goal_node} is not in 'goals' or was used already!")

    # After processing all paths, ensure none are left over
    if remaining_starts or remaining_goals:
        print(f"Not all starts/goals were used! " f"Remaining starts={remaining_starts}, goals={remaining_goals}")
    else:
        print("All starts and goals were matched with valid paths!")


def run_sparse_flow(dim=3, size=50, n_agents=10, seed=3):
    """
    Same logic as your 'main()', but returns the extracted paths.
    """
    t0 = time.perf_counter()

    # 1) Generate random starts/goals
    starts, goals, _ = generate_random_points(size, dim, n_agents, seed=seed)

    # 2) Build the grid adjacency matrix + super-source/sink
    A, node_count, s_idx, t_idx, all_coords = build_grid_matrix(size, dim, starts, goals)

    # 3) Solve for maximum flow (SciPy)
    res = maximum_flow(A, s_idx, t_idx)
    flow_matrix = res.flow

    # 4) Reconstruct paths
    paths = extract_paths_from_flow_matrix(flow_matrix, size, dim, starts, goals, super_source_idx=s_idx, super_sink_idx=t_idx)
    check_paths(starts, goals, paths)

    runtime = time.perf_counter() - t0
    plot_paths(starts=starts, goals=goals, paths=paths, dim=3, n_agents=len(paths))

    return paths, runtime


def solve_sparse_flow(starts, goals, dim=3, size=50):
    """
    Builds the adjacency matrix + super-source/sink, solves
    maximum flow using SciPy, extracts paths, and returns them.
    Returns (paths, runtime).
    """
    t0 = time.perf_counter()

    # 1. Build the adjacency matrix with super-source/sink
    A, node_count, s_idx, t_idx, all_coords = build_grid_matrix(size, dim, starts, goals)

    # 2. Solve for maximum flow
    res = maximum_flow(A, s_idx, t_idx)
    flow_value = res.flow_value
    flow_matrix = res.flow

    # 3. Extract paths
    paths = extract_paths_from_flow_matrix(flow_matrix, size, dim, starts, goals, super_source_idx=s_idx, super_sink_idx=t_idx)

    # Optional: check_paths(starts, goals, paths)

    runtime = time.perf_counter() - t0
    return paths, runtime


def main():
    """
    Original main() that just prints times, etc.
    """
    paths, runtime = run_sparse_flow()
    print(f"[sparse_flow] Found {len(paths)} paths in {round(runtime, 3)}s.")


if __name__ == "__main__":
    main()
