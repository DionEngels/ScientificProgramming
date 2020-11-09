# Function plot_graph is contributed by students vdWildenberg, vDort and Hendriks
#
# networkx is not standard part of the python distribution, install on ubuntu:
#
# sudo apt install python-pip
# sudo apt install python3-pip
# pip install networkx==2.2
# pip3 install networkx==2.2
#
# possibly, only the first time a figure shows up -- there are problems with the
# matplotlib backend on ubuntu ...
#
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib

# A: adjacency matrix, XYZ: vertex coordinates
def plot_graph(A, XYZ):
 # get I, J, V 
 I, J, _ = sparse.find(A)
 
 # create graph G from edges
 edges = zip(I.tolist(), J.tolist())
 G = nx.Graph()
 G.add_edges_from(edges)
 
 # generate vertex label dictionary
 size = max(max(I), max(J)) + 1
 vertex_labels = dict(zip(np.arange(size), np.arange(size)))
 
 # generate edge label dictionary
 edge_labels = dict( ((u, v), i) for i, (u, v) in enumerate(G.edges() ))
 # generate position dictionary
 pos = dict((i, j) for i,j in enumerate(XYZ.T))
 
 # draw edges
 nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
 # draw nodes
 nx.draw(G, pos=pos, labels=vertex_labels, with_labels=True)
    

# example
I = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7]
J = [0, 2, 4, 7, 0, 2, 5, 6, 0, 2, 4, 5, 6, 7, 0, 2, 3, 4, 2, 5, 6, 0, 4, 1, 5, 6]
V = np.ones(len(I), dtype=int)
C = sparse.coo_matrix((V, (I,J)))
XY = np.array([[0, 1, 2, 0, 1, 2, 0, 1],  [0, 0, 0, 1, 1, 1, 2, 2]])

print('C:\n'); print(C.toarray())
print('XY:\n'); print(XY)

plot_graph(C, XY)
