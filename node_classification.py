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
    x
    C

    Returns
    -------

    """
    b = [x]
    for c in C[x]:
        b.extend([i for i in branch(c, C)])
    return b


def full_node_classification(G):
    """
    procedure which does the full node classification of a networkx graph G
    
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph instance from networkx

    Returns
    -------
    graph_list: list
        list of all graphs for each iteration step
    non_roots: list
        list of non-root nodes in trees
    roots: list
        list of root nodes
    bulk: list
        list of nodes not in tree-shaped parts
    branch_dict: dict 
        dict of branches for all nodes
    children_dict: dict 
        dict of children for all nodes
    parents_dict: dict
        dict of parents for all tree-nodes and roots
    depth: dict
        dict of depth levels for all tree-nodes and roots
    height: dict
        dict of heights for all nodes
    branch_sizes: dict
        dict of branch sizes for all nodes

    """

    # return values
    graph_list = list() # 0 

    non_roots = set()  # 1
    roots = set()  # 2
    bulk = set()  # 3
    
    branch_dict = dict()  # 4
    children_dict = dict()  # 5
    parents_dict = dict()  # 6
    
    depth = dict()  # 7
    height = dict()    # 8
    branch_sizes = dict()   # 9
    
    # special case for tree graphs with odd diameter
    maxIter = np.infty
    if is_tree(G):
        diam = diameter(G)
        if diam % 2 == 1:
            maxIter = (diam - 1) / 2


    #### Identification of non-root nodes, parents, children, branches, and depths
    graph_list.append( G.copy() )
    # list of nodes for each level
    Vs = list()
    Vs.append( set(G.nodes()) )
    # list of leaves for each level
    Ds = list()
    # initialize branch, children and depth dicts
    branch_dict = { x:list() for x in Vs[0] }
    children_dict = { x:list() for x in Vs[0] }
    # iteration step    
    lvl = 0
    while True:
        # find all leaves of current level
        Ds.append( set( [ x for x, deg in degree(graph_list[lvl], Vs[lvl]).items() if deg==1] ) )
        # check if leaves not empty
        if (len(Ds[lvl]) == 0) or (lvl == maxIter):
            break
        # continue if leaves present
        else:
            # define nodes and graphs of next level
            Vs.append( Vs[lvl] - Ds[lvl] )
            graph_list.append( subgraph(graph_list[lvl], nbunch=Vs[lvl+1]))

            # add leaves to non-root nodes
            non_roots.update(Ds[lvl])

            # calculate further measures
            for x in Ds[lvl]:
                # add leaves to parent"s list of children
                parents_dict[x] = neighbors(graph_list[lvl], x)[0]
                children_dict[parents_dict[x]].append(x)
                # determine branch for each removed leave
                branch_dict[x] = branch(x,children_dict)
                # save depth of each leave
                depth[x] = lvl
            # increase level counter
            lvl+=1
    
    #### Identification of root nodes
    #determine all root nodes
    roots = set([ parents_dict[x] for x in non_roots]) - non_roots

    #determine branches and depth levels of roots
    for r in roots:
        # determine branch for roots
        branch_dict[r] = branch(r, children_dict)
        # calculate depth of root
        depth[r] = 1 + max([ depth[x] for x in children_dict[r] ])
             
    #calculate branch sizes for all nodes
    branch_sizes = { x: len(branch) for x,branch in branch_dict.items() }
    
    #### Identification of heights (this implementation is still a bit clumsy)
    Hs = list()
    Hs.append(roots)
    Ws = list()
    Ws.append(roots)
    i = 0 
    while len(Hs[i]) != 0:
        Hn = list()
        for x in Hs[i]:
            for c in children_dict[x]:
                if c not in Ws[i]:
                    Hn.append(c)
            # save the height value
            height[x]=i
        Hs.append( Hn )
        
        Wn = list()
        for l in Ws:
            for x in l:
                Wn.append(x)
        Ws.append( Wn )
    
        i+=1
        
    #### Identification of non-Tree nodes
    bulk = Vs[0] - roots - non_roots
    
    # return all determined values
    return graph_list, list(non_roots), list(roots), list(bulk), branch_dict, children_dict, parents_dict, depth, height, branch_sizes


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
    Gs, N, R, X, B, C, p, delta, eta, beta = full_node_classification(G)
    
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
    G = TestNetwork(200)
    cats = node_categories(G, denseThres=5)
    plot_network(G, cats)
    





