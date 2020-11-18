# ---
# jupyter:
#   jupytext:
#     formats: src/notebooks//ipynb,src/python//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Graphs
#
# ## 280. Example with plot

# + pycharm={"name": "#%%\n"}
import networkx as nx

nodes_pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
nodes_pos_double = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
arc_list = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]

G = nx.DiGraph()
G.add_nodes_from(nodes_pos.keys())

G.add_edges_from(arc_list)

# + [markdown] pycharm={"name": "#%% md\n"}
# It turns out that drawing self loops with `networkx` is impossible:

# + pycharm={"name": "#%%\n"}
# pos is a dictionary with the nodes (that we added before) as keys
# and their positions as values.
nx.draw(G, pos=nodes_pos)

# + [markdown] pycharm={"name": "#%% md\n"}
# Fortunately, we can use graphviz through `pygraphviz` to create png images (and show them inline in a notebook).

# + pycharm={"name": "#%%\n"}
# Requires graphviz (install with pacman on Arch, probably apt-get graphviz on ubuntu)
import pygraphviz as pgv
from networkx.drawing.nx_pydot import to_pydot
from networkx.drawing.nx_agraph import to_agraph
from IPython.display import Image

g_string = to_pydot(G).to_string()
Image(pgv.AGraph(g_string).draw(format='png', prog='dot'))


# + [markdown] pycharm={"name": "#%% md\n"}
# Unfortunately, graphviz (or the dot language) does not have a way to force the nodes to be drawn in a certain position, but networkx can do this.
# So depending on your use case:
#
# - Use `graphviz` when plotting self loops.
# - Use `networkx` when plotting a graph with nodes in certain positions.
#
# In all other cases, use what seems the easiest/most suitable.
# The `networkx` plots look more consistent with `matplotlib` plots, but `networkx` will deprecate drawing/plotting functionality at some point in the future and encourage tools like `graphviz` (which are made for drawing graphs).
#
# For drawing with `graphviz`, we can use the following function:

# + pycharm={"name": "#%%\n"}
def draw(g, path=None):
    """
    Draw the graph g and return as png. The graph is plotted inline in an IPython notebook.
    When a path is given, also save the image to a file at path, using the export function.
    """
    if path is not None:
        to_agraph(G).draw(format='png', prog='dot', path=path)
    return Image(to_agraph(G).draw(format='png', prog='dot'))

draw(G, 'test.png')
draw(G)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 295. Connectivity matrix example

# + pycharm={"name": "#%%\n"}
import numpy as np
import scipy.sparse as sparse

A = sparse.random(8, 8, 0.2, data_rvs=np.ones)
A.toarray()  # To print A

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 302. Incidence matrix

# + pycharm={"name": "#%%\n"}
import networkx.linalg.graphmatrix as gm
G = nx.gn_graph(5)
# oriented=True is needed to get the directed/oriented incidence matrix.
gm.incidence_matrix(G, oriented=True).toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 305. Incidence matrix

# + pycharm={"name": "#%%\n"}
n = 4
arc_list = [(1, 1), (1,2), (1,3), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4)]
G = nx.DiGraph()
G.add_nodes_from(range(1, n+1))
G.add_edges_from(arc_list)
gm.incidence_matrix(G, oriented=True).toarray()


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 306. Mathamatica GraphPlot[]
#
# Skipping this example for several reasons:
#
# 1. It is not very useful in the context of the course (I also don't remember that I needed this during the course).
# 2. There exist dedicated graph visualization libraries, like graphviz.
# 3. I don't have Mathematica installed and probably can't get to a license anymore.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 310. Hamiltonian path

# + pycharm={"name": "#%%\n"}
def plot_dodecahedral_graph(layout):
    G = nx.dodecahedral_graph()
    pos = layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    ham_cycle = list(zip(range(19), range(1, 20))) + [(19, 0)]
    nx.draw_networkx_edges(G, pos, edgelist=ham_cycle, edge_color='red', width=2)

plot_dodecahedral_graph(nx.kamada_kawai_layout)

# + [markdown] pycharm={"name": "#%% md\n"}
# The planar layout doesn't look good:

# + pycharm={"name": "#%%\n"}
plot_dodecahedral_graph(nx.planar_layout)

# + [markdown] pycharm={"name": "#%% md\n"}
# # Permutations
#
# ## 321. Permutations: representations

# + pycharm={"name": "#%%\n"}
A = sparse.random(8, 8, .3, format='csc', data_rvs=np.ones, dtype='uint8')
A.A  # Just found out that .A is equivalent with .toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# The following is equivalent to Matlab's `A(p, p)`:
# https://stackoverflow.com/a/45739482/6629569

# + pycharm={"name": "#%%\n"}
p = [3, 6, 0, 4, 1, 2, 7, 5]
I, J = np.ix_(p, p)
B = A[I, J]
B.A


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 322. Inverse permutation
#
# Define (linear-time) function to get the inverse permutation array.
# https://stackoverflow.com/a/25535723/6629569
#
# Apply it to B to check if we get A back:

# + pycharm={"name": "#%%\n"}
def inverse_permutation(p):
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q

q = inverse_permutation(p)
I, J = np.ix_(q, q)
B[I, J].A

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 323. Random permutation

# + pycharm={"name": "#%%\n"}
np.random.permutation(12)

# + pycharm={"name": "#%%\n"}
np.random.permutation(12)[:5]

# + [markdown] pycharm={"name": "#%% md\n"}
# # Reordering algorithms
#
# The test matrix used in the slides:

# + pycharm={"name": "#%%\n"}
nodes = range(1, 17)
edge_list = [(1, 2), (1, 5), (1, 6), (2, 3), (2, 6), (2, 7), (3, 4), (3, 7), (4, 7), (4, 8), (5, 6), (5, 9), (5, 10), (6, 10), (6, 7), (7, 11), (7, 8), (8, 11), (8, 12), (9, 10), (9, 13), (10, 13), (10, 14), (10, 11), (11, 15), (11, 16), (11, 12), (12, 16), (13, 14), (14, 15), (15, 16)]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)
nx.draw(G, with_labels=True)

# + [markdown] pycharm={"name": "#%% md\n"}
# It looks like `scipy.sparse.csgraph` has some of the reordering algorithms mentioned in the lecture notes.
# At least:
#
# ### 1. BFS levelset
# _This could also be 2. BFS queue, depending on the actual implementation?_
#
# Note that `i_start` is the index of the adjacency matrix.
# Thus, even though we have indexed our nodes starting at 1, in the adjacency matrix the indices start from 0 (as they always do in Python).

# + pycharm={"name": "#%%\n"}
import scipy.sparse.csgraph as csgraph

csgraph.breadth_first_order(gm.adjacency_matrix(G), i_start=0, return_predecessors=False)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### 4. Reverse Cuthill McKee

# + pycharm={"name": "#%%\n"}
csgraph.reverse_cuthill_mckee(gm.adjacency_matrix(G))

# + [markdown] pycharm={"name": "#%% md\n"}
# # Grids
#
# ## 450. Triangular grid

# + pycharm={"name": "#%%\n"}
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
triangles = np.array([[0, 1, 2], [1, 2, 3]])
plt.triplot(points[:, 0], points[:, 1], triangles)

# + pycharm={"name": "#%%\n"}
tri = Delaunay(points)
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
# -

# ## 451. Grid
#
# Little showcase to check if `trimesh` is working:
# (uncomment the `mesh.show()`, it is too heavy for PyCharm to keep it running...)

# + pycharm={"name": "#%%\n"}
import trimesh

mesh = trimesh.primitives.Sphere()
# mesh.show()

# + pycharm={"name": "#%%\n"}
vertices = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 2.0]])

# We need both sides of each face, otherwise the face will be invisible from one side.
# There are better ways to do this, but for this example it is good enough.
faces = [
    [0, 1, 2], [0, 2, 3], [0, 3, 1], [4, 1, 2], [4, 2, 3], [4, 1, 3],
    [0, 2, 1], [0, 3, 2], [0, 1, 3], [4, 2, 1], [4, 3, 1], [4, 3, 1]
]

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces
)

# mesh.show()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 461. Grid connectivity matrix
#
# Math walks of the slide :p
#
