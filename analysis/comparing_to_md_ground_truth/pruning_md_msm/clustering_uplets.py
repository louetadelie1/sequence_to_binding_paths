import itertools
import math
import random
from collections import Counter, defaultdict, OrderedDict
from itertools import combinations, islice
from math import nan, isnan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.linalg import expm, eig, norm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize
import networkx as nx
from community import community_louvain
import community as c  # You may only need one of these
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
import mdtraj as md
import pickle

def network_graph_microstates(filtered_merged_output, resolution=None, seed=42):
    """
    Build a network graph from transition data.

    Parameters
    ----------
    filtered_merged_output : list of tuples
        List of ((node1, node2), weight) edges.
    plot : bool, default False
        If True, draw the graph.
    resolution : float, default 3
        Resolution parameter for Louvain community detection.
    seed : int, default 42
        Random seed for layout and Louvain algorithm.

    Returns
    -------
    parts : dict
        Mapping of node -> community index.
    G : nx.Graph
        The graph object.
    pos : dict
        Layout positions of nodes.
    values : list
        Community index for each node in G.nodes().
    node_size : list
        Node sizes scaled by degree centrality.
    """
    G = nx.Graph()

    # Add edges with weights (force weak connections as weight=1)
    for edge, weight in filtered_merged_output:
        node1, node2 = edge
        if weight < 1:
            weight = 1
        G.add_edge(tuple(node1), tuple(node2), weight=weight)

    # Remove self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # Louvain community detection
    communities=nx.community.louvain_communities(G, seed=42,weight='weight',resolution=resolution) 
    parts = {node: i for i, community in enumerate(communities) for node in community}
    values = [parts.get(node) for node in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Node sizes (based on degree centrality)
    betCent = nx.degree_centrality(G)
    #node_size = [(v * 500)**2 for v in betCent.values()]

    return parts, G, pos, values,communities,betCent


def network_graph_macrostates(parts, P_eq_keys, communities, G, protein_name="Protein"):
    """
    Build and plot the macrostate-level graph from community clustering.

    Parameters
    ----------
    parts : dict
        Node -> community index mapping.
    P_eq_keys : dict
        Equilibrium populations for each node.
    communities : list of sets
        Microstate communities.
    G : nx.Graph
        The microstate graph.
    protein_name : str, default "Protein"
        Title for the plot.

    Returns
    -------
    community_centrality : dict
        Population per community.
    kd_centrality_ordered : dict
        Communities ranked by population.
    inv_map : dict
        Community index -> list of nodes.
    inv_map_vals : list of lists
        Values of inv_map as lists of nodes.
    pos : dict
        Node positions for plotting.
    """
    # Build inverse mapping community -> list of nodes
    inv_map = {}
    for k, v in parts.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    inv_map_vals = list(inv_map.values())

    # Compute community populations
    kd_centrality = {i: sum(P_eq_keys[node] for node in community) for i, community in enumerate(communities)}
    #highest_kd_community = max(kd_centrality, key=kd_centrality.get)
    kd_centrality_ordered = {k: v for k, v in sorted(kd_centrality.items(), key=lambda item: item[1][0], reverse=True)}

    # Layout for communities
    supergraph = nx.cycle_graph(len(inv_map_vals))
    superpos = nx.spring_layout(supergraph, scale=100, seed=2)

    pos = {}
    for center, comm in zip(superpos.values(), inv_map_vals):
        subgraph = G.subgraph(comm)
        subgraph_pos = nx.spring_layout(subgraph, center=center, seed=2)
        pos.update(subgraph_pos)

    # Define sigmoid scaling
    def sigmoid_magnification(val, A, k, x0):
        return A / (1 + math.exp(-k * (val - x0)))

    # # # Color cycle
    # color_cycle = itertools.cycle([plt.cm.Purples(i) for i in np.linspace(0, 1, len(inv_map_vals))])

    # # Draw nodes community by community
    # for i, (nodes, color) in enumerate(zip(inv_map_vals, color_cycle)):
    #     kd = [P_eq_keys[node] for node in nodes]
    #     sizes_shared = [sum(kd)] * len(nodes)
    #     sigmoid_values = [sigmoid_magnification(val, 1000, 200, 0.025) for val in sizes_shared]
    #     nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=[color], node_size=sigmoid_values, alpha=0.7, edgecolors=color)

    # # Draw edges
    # G.remove_edges_from(list(nx.selfloop_edges(G)))
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(False)
    # nx.draw_networkx_edges(G, pos=pos, alpha=0.1)

    # plt.title(protein_name)
    # plt.tight_layout()
    # plt.show()

    return kd_centrality_ordered, inv_map, inv_map_vals, pos
