import time
import random
import numpy as np
import networkx as nx
from itertools import product
from matplotlib import colors
import matplotlib.pyplot as plt


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


def build_grid_graph(dim, size):
    """
    Builds a directed grid graph of dimension 'dim' and 'size' in each axis.
    All edges have capacity=1, weight=1 (for the min-cost flow).
    """
    # Create undirected grid, then convert to directed
    uG = nx.grid_graph(dim=[size] * dim)
    G = nx.DiGraph(uG)

    # Assign default cost/weight/capacity to each edge
    nx.set_edge_attributes(G, 1, "capacity")
    nx.set_edge_attributes(G, 1, "weight")

    return G


def add_super_source_sink(G, starts, goals):
    """
    Adds a super-source (v_source) connecting to each start,
    and a super-sink (v_sink) to which each goal connects.
    Returns the labels of the super source and sink.
    """
    v_source, v_sink = "SOURCE", "SINK"
    G.add_node(v_source)
    G.add_node(v_sink)

    for s in starts:
        G.add_edge(v_source, s, capacity=1, weight=1)
    for g in goals:
        G.add_edge(g, v_sink, capacity=1, weight=1)

    return v_source, v_sink


def extract_paths_from_flow(G, flow_dict, starts, goals):
    """
    Given the flow dict from min-cost flow and the original starts/goals,
    reconstructs paths for each agent by 'following' the edges whose flow==1.
    Returns a list of paths (lists of nodes).
    Raises PathNotFound if any path is incomplete.
    """
    paths = []

    # Convert goals to a set for faster membership check
    goal_set = set(goals)

    for start in starts:
        path = []
        current_node = start

        # Reconstruct path by following edges with flow==1
        while True:
            path.append(current_node)
            if current_node in goal_set:
                # If this node is one of the goals, we're done with this path
                break

            # Find an outgoing edge with flow=1
            next_node = None
            for _, t in G.out_edges(current_node):
                if flow_dict[current_node][t] == 1:
                    next_node = t
                    break

            if next_node is None:
                # No valid flow out: path reconstruction failed
                path = []
                break
            current_node = next_node

        if not path or path[-1] not in goal_set:
            # We never reached a goal, or path is empty
            raise PathNotFound(f"No valid path found from start={start}")
        paths.append(np.asarray(path))

    return paths


import matplotlib.pyplot as plt
from matplotlib import colors


def plot_paths(paths, starts, goals, dim, n_agents, title="Multi-Agent Paths"):
    print("plot")
    """
    Plots the paths in either 2D or 3D with a single legend showing
    'Start', 'Path', 'Goal'. Each agent gets a unique color.

    paths  : List of np.ndarray (one array for each agent)
    starts : List of start positions (not used here, but typically the same as paths[i][0])
    goals  : List of goal positions (not used here, but typically the same as paths[i][-1])
    dim    : 2 or 3
    n_agents : number of agents (for color scaling)
    title  : title for the figure
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
            # Line for path
            path_objs = axes.plot(path[:, 0], path[:, 1], color=color, lw=2, label="Path")
            # Scatter goal
            goal_obj = axes.scatter(path[-1][0], path[-1][1], color=color, marker="*", s=200, label="Goal")

        else:  # dim == 3
            start_obj = axes.scatter(path[0][0], path[0][1], path[0][2], color=color, marker="x", s=200, label="Start")
            path_objs = axes.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2, label="Path")
            goal_obj = axes.scatter(path[-1][0], path[-1][1], path[-1][2], color=color, marker="*", s=200, label="Goal")

        # We only add "Start", "Path", and "Goal" to the legend once (for i=0)
        if i == 0:
            # path_objs is a list (from plt.plot), so extract the line handle
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


def run_nx_flow(dim=3, size=25, n_agents=10, seed=3):
    """
    Same logic as your 'main()', but returns the extracted paths.
    """
    t0 = time.perf_counter()

    # 1. Generate random starts/goals
    starts, goals, _ = generate_random_points(size, dim, n_agents, seed=seed)

    # 2. Build directed grid graph
    G = build_grid_graph(dim, size)

    # 3. Add super-source and super-sink
    v_source, v_sink = add_super_source_sink(G, starts, goals)

    # 4. Solve for min cost flow (NetworkX)
    flow_dict = nx.max_flow_min_cost(G, v_source, v_sink, capacity="capacity", weight="weight")

    # 5. Extract paths
    paths = extract_paths_from_flow(G, flow_dict, starts, goals)
    check_paths(starts, goals, paths)

    # Cleanup
    G.remove_node(v_source)
    G.remove_node(v_sink)

    runtime = time.perf_counter() - t0
    plot_paths(starts=starts, goals=goals, paths=paths, dim=3, n_agents=len(paths))

    return paths, runtime


def solve_nx_flow(starts, goals, dim=3, size=25):
    """
    Builds the Nx graph, adds super-source/sink, solves
    min-cost flow using NetworkX, extracts paths, and returns them.
    Returns (paths, runtime).
    """
    t0 = time.perf_counter()

    # 1. Build directed grid graph
    G = build_grid_graph(dim, size)

    # 2. Add super-source and super-sink
    v_source, v_sink = add_super_source_sink(G, starts, goals)

    # 3. Solve for min cost flow
    flow_dict = nx.max_flow_min_cost(G, v_source, v_sink, capacity="capacity", weight="weight")

    # 4. Extract paths
    paths = extract_paths_from_flow(G, flow_dict, starts, goals)

    # Optional: check_paths(starts, goals, paths)

    # 5. Cleanup
    G.remove_node(v_source)
    G.remove_node(v_sink)

    runtime = time.perf_counter() - t0
    return paths, runtime


def main():
    """
    Original main() that just prints times, etc.
    """
    paths, runtime = run_nx_flow()  # uses defaults
    print(f"[nx_flow] Found {len(paths)} paths in {round(runtime, 3)}s.")


if __name__ == "__main__":
    main()
