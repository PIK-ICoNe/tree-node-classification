# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:40:53 2016

@author: jannitzbon

This is an implementation of a node classification algorithm which aims at
systematically classifying nodes in  tree-like structures of networks.

[1] Nitzbon, J., Schultz, P., Heitzig, J., Kurths, J., & Hellmann, F. (2017). 
        Deciphering the imprint of topology on nonlinear dynamical network stability. 
        New Journal of Physics, 19(3), 33029. 
        https://doi.org/10.1088/1367-2630/aa6321
"""

import numpy as np
from networkx import average_neighbor_degree, degree, diameter, is_tree, neighbors, subgraph


def branch(x, C):
    """
    auxiliary function which returns the full branch of a node x given its children C
    
    Parameters
    ----------
    x: int
        node index
    C: dict
        dict of children for all nodes

    Returns
    -------
    b: list
        List of children in branch traversal starting at x.

    """
    b = [x]
    for c in C[x]:
        b.extend([i for i in branch(c, C)])
    return b

def calculate_heights(roots, children_dict):
    """
    
    Parameters
    ----------
    roots: set
        Set of root nodes as starting points for height counting
    children_dict: dict 
        dict of children for all nodes

    Returns
    -------
    height: dict
        height values for all tree-nodes and roots

    """
    height = dict()

    Hs = list()
    Hs.append(roots)
    Ws = list()
    Ws.append(roots)

    i = 0
    while len(Hs[i]) != 0:
        new_H = set()
        for x in Hs[i]:
            new_H.update(children_dict[x] - Ws[i])
            # save the height value
            height[x] = i
        Hs.append(new_H)

        new_W = set()
        for l in Ws:
            new_W.update(l)
        Ws.append(new_W)

        i += 1

    return height



def full_node_classification(G, debug=False):
    """
    procedure which does the full node classification of a networkx graph G
    
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph instance from networkx
    debug: boolean
        If true, give additional output for debugging.

    Returns
    -------
    tree_nodes: list
        list of non-root nodes in trees
    roots: list
        list of root nodes
    bulk: list
        list of nodes not in tree-shaped parts
    depth: dict
        dict of depth levels for all tree-nodes and roots
    height: dict
        dict of heights for all nodes
    nodes_per_level: list (optional)
        List of sets of nodes considered in each level of the iteration.
        The subgraph for each level is induced by the corresponding node set.
    branch_dict: dict (optional)
        dict of branches for all nodes
    children_dict: dict (optional)
        dict of children for all nodes
    parents_dict: dict (optional)
        dict of parents for all tree-nodes and roots
    branch_sizes: dict (optional)
        dict of branch sizes for all nodes

    """

    tree_nodes = set()
    parents_dict = dict()
    depth = dict()

    #### take care of special case for tree graphs with odd diameter
    maxIter = np.infty
    if is_tree(G):
        diam = diameter(G)
        if diam % 2 == 1:
            maxIter = (diam - 1) / 2

    # list of nodes for each level
    nodes_per_level = list()
    nodes_per_level.append(set(G.nodes()))

    # list of leaves for each level
    leaves_per_level = list()

    # initialize branch, children dicts
    branch_dict = {x: list() for x in nodes_per_level[0]}
    children_dict = {x: set() for x in nodes_per_level[0]}

    #### iteration
    lvl = 0
    while True:
        # induced subgraph
        graph = subgraph(G, nbunch=nodes_per_level[lvl])

        # find all leaves of current level
        leaves_per_level.append(set([x for x, deg in degree(graph, nodes_per_level[lvl]).items() if deg==1]))

        # check if leaves not empty
        if (len(leaves_per_level[lvl]) == 0) or (lvl == maxIter):
            break
        # continue if leaves present
        else:
            # define nodes and graphs of next level
            nodes_per_level.append( nodes_per_level[lvl] - leaves_per_level[lvl] )

            # add leaves to non-root nodes
            tree_nodes.update(leaves_per_level[lvl])

            # update list of parents
            parents_dict.update({x: neighbors(graph, x)[0] for x in leaves_per_level[lvl]})

            for x in leaves_per_level[lvl]:
                # add leaves to parent"s list of children
                children_dict[parents_dict[x]].add(x)
                # determine branch for each removed leaf
                branch_dict[x] = branch(x, children_dict)
                # save depth of each leaf
                depth[x] = lvl
            # increase level counter
            lvl+=1

    #### determine all root nodes
    roots = set(map(parents_dict.get, tree_nodes)) - tree_nodes

    #### determine branches and depth levels of roots
    for r in roots:
        # determine branch for roots
        branch_dict[r] = branch(r, children_dict)
        # calculate depth of root
        depth[r] = 1 + max([depth[x] for x in children_dict[r]])

    #### calculate branch sizes for all nodes
    branch_sizes = {x: len(branch) for x, branch in branch_dict.items()}
    
    #### Identification of heights (this implementation is still a bit clumsy)
    height = calculate_heights(roots, children_dict)
        
    #### Identification of non-Tree nodes
    bulk = nodes_per_level[0] - roots - tree_nodes

    if debug:
        return list(tree_nodes), list(roots), list(bulk), depth, height, nodes_per_level, branch_dict, children_dict, parents_dict, branch_sizes
    else:
        return list(tree_nodes), list(roots), list(bulk), depth, height



def node_categories(G, denseThres=5):
    """
    procedure which returns a dict of node categories indexed by the node number
    
    The categories are defined as in Figure 1 of [1]
        
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph instance from networkx
    denseThres: int
        Degree limit between dense and sparse sprouts

    Returns
    -------
    cat: dict
        Dictionary assigning a category string to each node.
        

    """
    N, R, X, delta, eta = full_node_classification(G, debug=False)
    
    avndeg = average_neighbor_degree(G)
    cat = dict()
    
    for x in G.nodes():
         if x in X: cat[x] = "bulk"
         if x in R: cat[x] = "root"
         if x in N: 
             cat[x] = "inner tree node"
             #leaves
             if delta[x] == 0:
                 cat[x] = "proper leaf"
                 #sprouts
                 if eta[x] == 1:
                     cat[x] = "sparse sprout"
                     if avndeg[x] >= denseThres:
                         cat[x] = "dense sprout"
    
    return cat

def TestNetwork(n=42, p=0.2):
    """
    
    Parameters
    ----------
    n: int
        Number of nodes
    p: float
        Distance threshold value

    Returns
    -------
    Gmst: networkx.classes.graph.Graph
        Graph instance of networkx, spatially embedded minimum spanning tree with one additional edge.

    """
    from networkx import random_geometric_graph, is_connected, get_node_attributes, minimum_spanning_tree
    import random

    #assert n >= 20

    # fix random seed to obtain reproducable networks
    random.seed(42)

    # generate connected random geometric graph Gtest
    while True:
        Gtest = random_geometric_graph(n, p)
        if is_connected(Gtest):
            break
    pos = get_node_attributes(Gtest, "pos")

    # generate minimum spanning tree Gmst
    Gmst = minimum_spanning_tree(Gtest)

    # add some arbitrary edge such that we have a cylce
    Gmst.add_edge(0, Gmst.number_of_nodes() - 1)

    return Gmst

def plot_network(G, cat):
    """
    
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph instance of networkx
    cat: dict
        Dictionary assigning a category string to each node.

    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot

    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from networkx import draw_networkx_nodes, get_node_attributes, draw_networkx_edges

    fig, ax = plt.subplots(1, 1, figsize=(11.7, 8.27))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.axis("off")

    # node positions
    pos = get_node_attributes(G, "pos")

    def node_color(n):
        ncd = {
            "bulk": "darkgray",
            "root": "saddlebrown",
            "sparse sprout": "aqua",
            "dense sprout": "royalblue",
            "proper leaf": "gold",
            "inner tree node": "limegreen"
        }
        return ncd.get(cat[n], "white")

    draw_networkx_nodes(G=G, pos=pos, ax=ax, node_color=[node_color(n) for n in G.nodes_iter()],
                                       node_shape="o", node_size=300, with_labels=False)
    draw_networkx_edges(G=G, pos=pos, ax=ax, width=2, edge_color="k", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()

    handles.append(mp.Patch(color="darkgray"))
    handles.append(mp.Patch(color="saddlebrown"))
    handles.append(mp.Patch(color="limegreen"))
    handles.append(mp.Patch(color="aqua"))
    handles.append(mp.Patch(color="royalblue"))
    handles.append(mp.Patch(color="gold"))
    labels.append("bulk")
    labels.append("root")
    labels.append("inner tree node")
    labels.append("sparse sprout")
    labels.append("dense sprout")
    labels.append("proper leaf")

    ax.legend(handles, labels, loc=0, fontsize=14, fancybox=True, markerscale=0.8, scatterpoints=1)

    fig.tight_layout()

    plt.show()

    return fig, ax


if __name__ == "__main__":
    G = TestNetwork(42)
    cats = node_categories(G, denseThres=5)
    plot_network(G, cats)
    





