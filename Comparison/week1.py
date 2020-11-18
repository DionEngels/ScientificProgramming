# ---
# jupyter:
#   jupytext:
#     formats: src/notebooks//ipynb,src/python//py
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

# + pycharm={"name": "#%%\n"}
import sys
import numpy as np
import scipy.sparse as sp
# -

# ## 23. Graph example with plot

# + pycharm={"name": "#%%\n"}
import networkx as nx
G = nx.Graph()
G.add_edges_from([(1, 1), (1, 2), (2, 3), (3, 1), (1, 4)])
nx.draw(G, with_labels=True)

# + [markdown] pycharm={"name": "#%% md\n"}
# # Functions and scripts
#
# ## 51. @-functions -> lambda functions
#
# When presenting this to students I think it should be made clear that you would only use a lambda function if you need the function right now, i.e., you don't usually define a function with a lambda function; you would use a regular function (using `def`).
# Examples on slide 68 should make that clear I think, but it's something to be aware of.

# + pycharm={"name": "#%%\n"}
a = 2
f = lambda x: a * x*x
f(np.array([3, 4]))  # f([3, 4]) gives TypeError: can't multiply sequence by non-int of type 'list'
# -

# Similar to MATLAB: lambda function can return only one value.
# (Try to execute the following: That call should return: 
# <div class="alert-danger">TypeError: 'tuple' object is not callable ...</div>

# + pycharm={"name": "#%%\n"}
g = lambda x: x, 2
g(3)


# + pycharm={"name": "#%%\n"}
def h(x):
    return x, 2
h(3)
# -

# ## 52. @-functions examples (magic)

# + pycharm={"name": "#%%\n"}
v = np.array([6, 1, 3, 2, 4, 2, 4, 5])
# -

# Note: `map` is a standard function in Python, returning a map object.
# This map object has to be transformed to a list with `list(.)` (pure Python) **and** then with `np.array(.)` (numpy/scipy).

# + pycharm={"name": "#%%\n"}
np.array(list(map(lambda x: x*x, v)))
# -

# In Python, it is more common to use a [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions), and cast the list to a numpy array.
# The running time is more or less the same [[SO](https://stackoverflow.com/a/46470401/6629569)].
# For squaring each element in a list, note that it's fastest (and shortest) to write simply `v**2`.

# + pycharm={"name": "#%%\n"}
np.array([x**2 for x in v])
# -

# Omiting the `np.array(.)` call for clarity, for the rest of the examples.

# + pycharm={"name": "#%%\n"}
[x**2 for x in []]

# + pycharm={"name": "#%%\n"}
max(v)

# + pycharm={"name": "#%%\n"}
min(v)

# + pycharm={"name": "#%%\n"}
sorted(v)

# + pycharm={"name": "#%%\n"}
np.exp(1)


# + pycharm={"name": "#%%\n"}
def g(x):
    return x*x

[g(x) for x in v]

# -

# The following also works:

# + pycharm={"name": "#%%\n"}
g(v)
# -

# (small example to show that the MATLAB example also works in Python, but you would use a (local) function definition instead of a lambda function.)

# + pycharm={"name": "#%%\n"}
list(map(g, v))


# + pycharm={"name": "#%%\n"}
def h(x):
    return -x

def f(x):
    return np.multiply(h(x), x>4) + np.multiply(g(x), x <= 4)

f(v)
# -

# # Numbers
#
# ## 66, 69. Decimal to binary

# + pycharm={"name": "#%%\n"}
np.binary_repr(6)

# + pycharm={"name": "#%%\n"}
np.binary_repr(-6)
# -

# ## 67. Real numbers
#
# 3. max real IEEE 64 bit (same as matlab) $1.7976 \cdot 10^{308}$

# + pycharm={"name": "#%%\n"}
sys.float_info
# -

# max real 128 bit $1.1897 \cdot 10^{4932}$, from:

# + pycharm={"name": "#%%\n"}

np.finfo(np.float128)
# -

# 4. Force overflow. Inserting one 0 more also gives M (what could be the reason ...?)

# + pycharm={"name": "#%%\n"}
M = np.finfo(np.float128).max
1.000000000000001 * M
# -

# ## 70. Smallest (positive) computer real

# + pycharm={"name": "#%%\n"}
for i in np.arange(46, 55 + 1):
    a = 1 + np.power(2, -np.float(i))
    print(f"(1 + 2^-{i}) - 1 = {a - 1}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 67,71. Largest computer numbers (int)
#
# From [numpy docs](https://numpy.org/devdocs/user/basics.types.html): $9223372036854775807$
#

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 73. Real number computer representation
#
# A very clear (infamous?) example of computer real-value round-off of errors:

# + pycharm={"name": "#%%\n"}
0.1 + 0.2 == 0.3

# + [markdown] pycharm={"name": "#%% md\n"}
# Examples from the slides:

# + pycharm={"name": "#%%\n"}
(1/10) * 7 - (7/10)

# + pycharm={"name": "#%%\n"}
(1/3) * 7 - (7/3)

# + pycharm={"name": "#%%\n"}
np.sin(np.pi / 4) - np.sqrt(2) / 2

# + pycharm={"name": "#%%\n"}
np.log(np.power(np.exp(1), 2)) - 2


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 77. Round-off consequences
#
# Checking if matrix is symmetrical with numpy: `np.allclose(A, A.T, rtol=0, atol = 0, equal_nan=False)`.
# With scipy (for sparse matrices) `(abs(A - A.T) > tol).nnz == 0`.
#
# Because numpy and scipy use real valued computer numbers round off is likely to occur. An exception are matrices such as B below which only contain (smaller) integer values (in real value computer number representation).

# + pycharm={"name": "#%%\n"}
def is_symmetric(A, tolerance=1e-16):
    return (abs(A - A.T) > tolerance).nnz == 0

n = 7

diagonals = [-1 * np.ones(n-1), 2 * np.ones(n), -1 * np.ones(n-1)]
B = sp.diags(diagonals, offsets=[-1, 0, 1])
is_symmetric(B)

# non-symmetric with non-integers
bdiagonals = [-1 * np.ones(n-1), 2 * np.ones(n)]
BB = sp.diags(bdiagonals, offsets=[-1, 0])
D = np.diag(np.array(range(n)))/n
# np.allcose(BB*D + D*BB, (BB*D + D*BB).T,rtol=0,atol=0) # needs a very recent numpy ...

# + pycharm={"name": "#%%\n"}
D = sp.diags(np.random.default_rng().normal(size=n))
is_symmetric(D)

# + pycharm={"name": "#%%\n"}
is_symmetric(D * B * D)

# + [markdown] pycharm={"name": "#%% md\n"}
# # Instructions
#
# ## 83. Instructions
#
# Non control instructions (same as on the slide)

# + pycharm={"name": "#%%\n"}
max(1, 2)

# + pycharm={"name": "#%%\n"}
v = [3, 2, 1]

# + pycharm={"name": "#%%\n"}
max(v)

# + pycharm={"name": "#%%\n"}
sorted(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# Control instructions
#
# First we define `x` so we can use it.
# Note that the default `if` statement cannot be written on one line, but it can as follows (alike the C construct "(x == 0)? 1 : 0":

# + pycharm={"name": "#%%\n"}
x = 1
1 if x == 0 else 0  # This actually returns a value

# + [markdown] pycharm={"name": "#%% md\n"}
# The default `if` statement:

# + pycharm={"name": "#%%\n"}
# This does not return a value (but we use a print statement to print it)
if x == 0:
    print(1)
else:
    print(0)

# + pycharm={"name": "#%%\n"}
for i in [1, 2]:
    print(i)

# + pycharm={"name": "#%%\n"}
while x < 2:
    print(x)
    x = x + 1

# + [markdown] pycharm={"name": "#%% md\n"}
# # Vectors
#
# ## 93. Instructions

# + pycharm={"name": "#%%\n"}
v = np.random.randint(1, 9, 12)
v

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
plt.plot(v)

# + pycharm={"name": "#%%\n"}
np.linalg.norm(np.zeros(3) - np.ones(3), ord=np.inf)

# + pycharm={"name": "#%%\n"}
v[6] = -1
v

# + [markdown] pycharm={"name": "#%% md\n"}
# Watch out when creating sets.
# In general, `{...}` and `set(..)` is the same, except when creating an empty set.
# `{}` creates an empty dictionary instead:

# + pycharm={"name": "#%%\n"}
print(type(set([1, 2])))
print(type({1,2}))
print(type(set()))
print(type({}))

# + pycharm={"name": "#%%\n"}
# setdiff
{2, 1, 1, 4} - set([5, 4, 6])

# + pycharm={"name": "#%%\n"}
# union
{2, 1}.union({3, 2, 4})

# + pycharm={"name": "#%%\n"}
# symmetric difference
{2, 1, 1, 4, 3}.symmetric_difference({5, 4, 3, 6})

# + pycharm={"name": "#%%\n"}
{1}.intersection({2, 1})

# + pycharm={"name": "#%%\n"}
v > 5

# + pycharm={"name": "#%%\n"}
v[v > 5]

# + [markdown] pycharm={"name": "#%% md\n"}
# `np.bincount` is similar to MATLAB's `accumarray` but it cannot handle negative numbers.

# + pycharm={"name": "#%%\n"}
v[6] = 3
np.bincount(v)

# + pycharm={"name": "#%%\n"}
np.unique(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# Note that adding lists in Python appends one list to the other, but in numpy this does an element-wise addition.
# This means that numpy cannot add lists of different size.
# This is similar when multiplying two vectors, or when multiplying with a scalar.

# + pycharm={"name": "#%%\n"}
[1, 2] + [4, 3, 5]

# + pycharm={"name": "#%%\n"}
np.array([1,2]) + np.array([4, 3])

# + pycharm={"name": "#%%\n"}
2 * [1, 2]

# + pycharm={"name": "#%%\n"}
2 * np.array([1, 2])

# + [markdown] pycharm={"name": "#%% md\n"}
# Multiplying python lists gives an error:

# + pycharm={"name": "#%%\n"}
[2, 3] * [3, 4]

# + [markdown] pycharm={"name": "#%% md\n"}
# Multiplying numpy lists gives the element-wise product (i.e., the hadamard product of two equal length arrays)

# + pycharm={"name": "#%%\n"}
np.array([2, 3]) * np.array([1, 2])


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 98/99. Vector creation and timings

# + pycharm={"name": "#%%\n"}
def loop(n):
    v = np.array([])
    for i in range(n):
        v.__add__(i)

def allocate(n):
    v = np.zeros(n)
    for i in range(n):
        v[i] = i


# + pycharm={"name": "#%%\n"}
import localtiming
n = [10000, 100000, 1000000, 10000000]

times = [[
    localtiming.measure(i, lambda x: range(x)),
    localtiming.measure(i, loop),
    localtiming.measure(i, allocate)
  ] for i in n]


# + pycharm={"name": "#%%\n"}
import pandas as pd
df = pd.DataFrame(times, columns=["range", "loop", "preallocated loop"], index=n)
df.plot.line(loglog=True, style='o-', grid=True)
plt.xlabel("vector length")
plt.ylabel("vector creation time")

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 122/123. Unique/sorting

# + pycharm={"name": "#%%\n"}
H = np.array([7, 4, 4, 1, 5, 2, 3, 2, 9, 4, 6, 3, 7, 1]).reshape((-1, 2))
H

# + pycharm={"name": "#%%\n"}
# https://stackoverflow.com/a/2828121/6629569
# Use -1 to reverse order
indices = np.lexsort((-H[:,0], -H[:,1]))
H[indices]

# + pycharm={"name": "#%%\n"}
_, i = np.unique(H[:,1], return_index=True)
H[i]

# + [markdown] pycharm={"name": "#%% md\n"}
# Using the [numpy indexed](https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP) package which claims to be vectorised and built upon using `np.argsort`.

# + pycharm={"name": "#%%\n"}
import numpy_indexed
_, max = numpy_indexed.group_by(H[:,1]).max(H)
max

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 128. Strings

# + pycharm={"name": "#%%\n"}
str.split("let's split this string")

# + pycharm={"name": "#%%\n"}
"concatenate" + "strings"

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 131. Cell-arrays
# Matlab cell arrays are regular Python lists -- conceptually!

# + pycharm={"name": "#%%\n"}
c = ['a', 1, 'b']

# + pycharm={"name": "#%%\n"}
c = [['club', ['seven', 'lady']], ['heart', ['two', 'king']]]

# -


