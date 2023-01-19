import time
import random
import numpy as np
import networkx as nx
from itertools import product
from matplotlib import colors
import matplotlib.pyplot as plt

# random.seed(3)
DIM = 3
start_time = []
start_time.append(time.perf_counter())


def timer(arg="Task"):
    start_time.append(time.perf_counter())
    print(f"{len(start_time)-1}.Time taken to do {arg} : {round(start_time[-1] - start_time[-2], 3)}s.")


def main():
    dim = DIM
    size = 20
    n_agents = 20  # number of agents
    print(f"n_agents : {n_agents}")

    nodes = list(product(*[range(size)] * dim))
    inside_nodes = list(product(*[range(2, size - 2)] * dim))
    edge_nodes = [point for point in nodes if point not in inside_nodes]

    starts = [tuple(random.choice(inside_nodes))for _ in range(n_agents)]
    goals = [tuple(random.choice(edge_nodes))for _ in range(n_agents)]
    timer("Genarate start and goal locations")

    uG = nx.grid_graph(dim=[size for _ in range(dim)])
    G = nx.DiGraph(uG)
    print(f"  Undirected Grpah : {uG}")
    print(f"  Directed Grpah : {G}")
    timer("Genarate grid")

    for edge in G.edges:
        G.edges[edge]['weight'] = 1
        G.edges[edge]['capacity'] = 1
        G.edges[edge]['flow'] = 0

    v_source = 1
    v_sink = -1
    G.add_node(v_source)
    G.add_node(v_sink)
    for start, goal in zip(starts, goals):
        G.add_edge(v_source, start)
        G.edges[v_source, start]['capacity'] = 1
        G.edges[v_source, start]['weight'] = 1
        G.add_edge(goal, v_sink)
        G.edges[goal, v_sink]['capacity'] = 1
        G.edges[goal, v_sink]['weight'] = 1

    mincostFlow = nx.max_flow_min_cost(G, v_source, v_sink, capacity='capacity', weight='weight')
    mincost = nx.cost_of_flow(G, mincostFlow)
    timer("Solve for flow")
    print(f"  mincost = {mincost}")

    in_edges = [edge for edge in G.in_edges(v_sink)]
    out_edges = [edge for edge in G.out_edges(v_source)]
    in_flow = [mincostFlow[edge[0]][edge[1]] for edge in in_edges]
    out_flow = [mincostFlow[edge[0]][edge[1]] for edge in out_edges]

    if sum(in_flow) != n_agents or sum(out_flow) != n_agents:
        print("  There will be no path scenario")
        print(sum(in_flow), sum(out_flow), n_agents)

    paths = []
    for idx, start in enumerate(starts):
        path = []
        current_node = start

        while True:
            path.append(current_node)
            if current_node in goals:
                break

            out_edges = [edge for edge in G.out_edges(current_node)]
            out_flow = [mincostFlow[edge[0]][edge[1]] for edge in out_edges]

            if True in (flow == 1 for flow in out_flow):
                current_node = out_edges[out_flow.index(1)][1]
            else:
                path = []
                break

        paths.append(np.asarray((path)))
    path_edges = np.asarray(paths, dtype=object)

    G.remove_node(v_source)
    G.remove_node(v_sink)
    G.remove_nodes_from(list(nx.isolates(G)))

    fig = plt.figure(f"Multi Agent Path Planner")
    if dim == 2:
        axes = fig.add_subplot()
    if dim == 3:
        axes = fig.add_subplot(projection="3d")
        axes.set_zlabel("Z Axis")
    axes.set_xlabel("X Axis")
    axes.set_ylabel("Y Axis")
    axes.set_title(f"{dim}D plot of  Paths")
    cn = colors.Normalize(0, n_agents)
    for i, path in enumerate(path_edges):
        if dim == 2:
            if len(path) > 2:
                axes.plot(path[:, 0], path[:, 1], color=plt.cm.jet(cn(i)), lw=2, alpha=1)
                #axes.scatter(path[:, 0], path[:, 1], color=plt.cm.jet(cn(i)), alpha=1)
            axes.scatter(path[0][0], path[0][1], color=plt.cm.jet(cn(i)), marker='x', s=200, alpha=1, label=f"Start {i}")
            axes.scatter(path[-1][0], path[-1][1], color=plt.cm.jet(cn(i)), marker='*', s=200, alpha=1, label=f"Goal {i}")
        if dim == 3:
            if len(path) > 2:
                axes.plot(path[:, 0], path[:, 1], path[:, 2], color=plt.cm.hsv(cn(i)), lw=2, alpha=1)
                #axes.scatter(path[:, 0], path[:, 1], path[:, 2], color=plt.cm.hsv(cn(i)), alpha=1)
            axes.scatter(path[0][0], path[0][1], path[0][2], color=plt.cm.hsv(cn(i)), marker='x', s=200, alpha=1, label=f"Start {i}")
            axes.scatter(path[-1][0], path[-1][1], path[-1][2], color=plt.cm.hsv(cn(i)), marker='*', s=200, alpha=1, label=f"Goal {i}")
    # plt.legend()
    timer("plotting graph")
    print(f"{len(start_time)}.Time taken total: {time.perf_counter() - start_time[0]}s ")
    plt.show()


if __name__ == '__main__':
    main()
