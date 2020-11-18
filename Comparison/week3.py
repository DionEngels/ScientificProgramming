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

import numpy as np
import scipy as sp
from scipy import linalg
from scipy import sparse

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 232. Example with CSC matrix

# + pycharm={"name": "#%%\n"}
A = sparse.diags([5, -1, 2], [0, -1, 1], (8, 8))
A.todense()

# + [markdown] pycharm={"name": "#%% md\n"}
# This matrix is stored in diagonal format by default:
#

# + pycharm={"name": "#%%\n"}
A

# + pycharm={"name": "#%%\n"}
sparse.find(A)

# + [markdown] pycharm={"name": "#%% md\n"}
# To create a matrix in another format, use `format='csc'`, where `csc` means the CCS format and can be replaced by another format.
# Note that Scipy uses the name Compressed Sparse Column format (instead of Compressed Column Sparse format).
# See the [Scipy docs on sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html) for the other names.

# + pycharm={"name": "#%%\n"}
B = sparse.diags([1, 2], [0, 1], (4, 4), format='csc')
B

# + pycharm={"name": "#%%\n"}
B.todense()

# + pycharm={"name": "#%%\n"}
print(f"Data (V): \n  {B.data}")
print(f"Row indices: \n  {B.indices}")
print(f"Indices (pointers) to column starts in data and row indices: \n  {B.indptr}")

# + [markdown] pycharm={"name": "#%% md\n"}
# To get the values of the nonzero entries, use `A.data`:
# (`A.nonzero()` gives the row and column indices of the nonzero entries)

# + pycharm={"name": "#%%\n"}
A.data

# + pycharm={"name": "#%%\n"}
data = [1, 2, 3, 4]  # V
row = [0, 1, 0, 2]  # I
column = [1, 3, 0, 0]  # J
A = sparse.csc_matrix((data, (row, column)), shape=(5,5))
A.todense()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 235. `spconvert()` and 237. `sparse()`
#
# SciPy also adds values in case of multiple index occurrences, when using the `(data, (row, column))` syntax to create a sparse (csc) matrix.

# + pycharm={"name": "#%%\n"}
sparse.csc_matrix(([-7, 2, 1], ([0, 0, 1], [0, 0, 1]),)).todense()


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 239. Complexity
#
# Only providing Python syntax for the instructions in the table.
# The complexity should be roughly the same as in Matlab but efficiency (speed) may be very different.
#
# Of course, import SciPy's sparse module with `from scipy import sparse`.
#
# #### Creation
# Use `sparse.<format>_matrix()` to create a matrix in the `<format>` format.
# E.g., to create a CSC matrix:

# + pycharm={"name": "#%%\n"}
I = [0, 0, 1, 2]
J = [0, 1, 1, 3]
V = [1, 2, 3, 4]
n, m = 4, 4
A = sparse.csc_matrix((V, (I, J)), (n, m))
A.todense()

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Select row
# Note that this is different from Matlab's `A(k)`.

# + pycharm={"name": "#%%\n"}
A[0].todense()

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Entry selection

# + pycharm={"name": "#%%\n"}
A[0, 1]

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Scalar multiplication

# + pycharm={"name": "#%%\n"}
(0 * A).todense()

# + pycharm={"name": "#%%\n"}
(3 * A).todense()
# -

# #### Multiplication

# + pycharm={"name": "#%%\n"}
x = [1, 2, 3, 4]
A.dot(x)

# + pycharm={"name": "#%%\n"}
A * x

# + pycharm={"name": "#%%\n"}
A @ x

# + pycharm={"name": "#%%\n"}
diag = sparse.spdiags(x, 0, len(x), len(x))
(A @ diag).toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Solve diagonal

# + pycharm={"name": "#%%\n"}
from scipy.sparse import linalg as splinalg
splinalg.spsolve_triangular(diag, x)

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Solve lower triangular

# + pycharm={"name": "#%%\n"}
L = sparse.tril(np.random.randint(0, 3, (4, 4)), k=-1) + sparse.eye(4)
splinalg.spsolve_triangular(L, x, unit_diagonal=True)

# + pycharm={"name": "#%%\n"}
#### Solving upper triangular

# + pycharm={"name": "#%%\n"}
U = sparse.triu(np.random.randint(0, 3, (4, 4)), k=1) + sparse.eye(4)
splinalg.spsolve_triangular(U, x, lower=False, unit_diagonal=True)

# + [markdown] pycharm={"name": "#%% md\n"}
# #### Scalar addition
# Matrix is not sparse anymore, so might as well convert to dense matrix first.
# (sparse matrices don't even support scalar addition)

# + pycharm={"name": "#%%\n"}
A.todense() + 3

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 240. Sources
# To print all matrices, `toarray()` was included everywhere

# + pycharm={"name": "#%%\n"}
sparse.coo_matrix((3,3)).toarray()

# + pycharm={"name": "#%%\n"}
sparse.eye(5, 7).toarray()

# + pycharm={"name": "#%%\n"}
sparse.random(5, 5, 0.3).toarray()

# + pycharm={"name": "#%%\n"}
sparse.diags([-1, 4, -1], [-1, 0, 1], (4, 4)).toarray()

# + pycharm={"name": "#%%\n"}
# This is brilliant!
A = sparse.diags([-1, 2, -1], [-1, 0, 1], (4, 4))
sparse.kron(sparse.eye(4, 4), A) + sparse.kron(A, sparse.eye(4, 4))


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 242. Zero loops

# + pycharm={"name": "#%%\n"}
def zero_loops(n):
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], (n, n))
    return sparse.kron(sparse.eye(n, n), A) + sparse.kron(A, sparse.eye(n, n))

zero_loops(3).toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 243. One loop
# `one_loop(n)` gets as input the size of a subblock, and creates an $$n^2 \times n^2$$ matrix of subblocks of size $$n \times n $$.

# + pycharm={"name": "#%%\n"}
import math

def one_loop(n):
    # assert math.sqrt(n).is_integer(), "n should be a square"
    # Use a dok matrix so we can easily assign elements.
    A = sparse.dok_matrix((n**2, n**2))
    # m = int(math.sqrt(n))
    for i in range(n**2):
        k = i % n + 1
        l = math.floor(i / n) + 1
        A[i, i] = 4
        if k > 1:
            A[i, i-1] = -1
        if k < n:
            A[i, i+1] = -1
        if l > 1:
            A[i, i-n] = -1
        if l < n:
            A[i, i+n] = -1
    return A.tocsc()

one_loop(3).toarray()


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 245. Two loops
# Skipping one loop with preallocation, because I am not sure if it really is preallocation?
# From what I understand from the [matlab documentation](https://nl.mathworks.com/help/matlab/ref/sparse.html), `sparse(n, n, 5*n)` creates a sparse matrix with one entry with value `5*n` in position `(n, n)`.

# + pycharm={"name": "#%%\n"}
def two_loops(n):
    A = sparse.dok_matrix((n**2, n**2))
    for k in range(n**2):
        for l in range(n**2):
            if l == k: A[k, l] = 4
            if l == k - 1 and k % n != 0: A[k, l] = -1
            if l == k + 1 and k % n != 2: A[k, l] = -1
            if l == k - n: A[k, l] = -1
            if l == k + n: A[k, l] = -1
    return A

two_loops(3).toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 247. Timings

# + pycharm={"name": "#%%\n"}
from timeit import timeit

# 100, 1000, 10000
# created matrix is of size n^2 x n^2
for i in [10, 31, 100]:
    n = int(i)
    print(timeit(
        f"zero_loops({n})",
        setup="from __main__ import zero_loops",
        number=5
    ))

# + pycharm={"name": "#%%\n"}
for i in [10, 31, 100]:
    n = int(i)
    print(timeit(
        f"one_loop({n})",
        setup="from __main__ import one_loop",
        number=5
    ))

# + pycharm={"name": "#%%\n"}
for i in [10, 31, 100]:
    n = int(i)
    print(timeit(
        f"two_loops({n})",
        setup="from __main__ import two_loops",
        number=5
    ))

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
plt.loglog([100, 31**2, 10000],
           np.array([0.010975192999467254, 0.011837242999718, 0.023965784000210988]) / 5,
           label="0 loops")

plt.loglog([100, 31**2, 10000],
           np.array([0.03488252199986164, 0.2660015530000237, 2.8797298780000347]) / 5,
           label="1 loop")

plt.loglog([100, 31**2, 10000],
           np.array([0.04005085099925054, 1.1138713059999645, 96.31783088499924]) / 5,
           label="2 loops")

plt.legend()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 250. Write to LaTeX
# I found the `array_to_latex` package, but there could be better options.
# It seems to work quite neat.

# + pycharm={"name": "#%%\n"}
import array_to_latex as a2l

B = zero_loops(3)
a2l.to_ltx(B.toarray(), frmt='{:1.0f}')

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 253. semilogy
# Reconsider: What is the MF3 matrix? Reproduce it with that particular matrix if necessary.

# + pycharm={"name": "#%%\n"}
A = sparse.random(100, 100, 0.2)
plt.semilogy(np.sort(A.data))

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 254. spy

# + pycharm={"name": "#%%\n"}
A = sparse.random(10, 10, 0.1)
plt.spy(A)
plt.spy(sparse.eye(10, 10), marker='o', color='red')
# Could not find a flipud for sparse matrices, but maybe didn't look good enough.
plt.spy(np.flipud(A.toarray()), marker='+', color='green')

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 255. Substitution

# + pycharm={"name": "#%%\n"}
B = sparse.diags([5, -1, 2], [1, 0, -1], (5, 5), format="csc")
B[0, 3] = 8; B[3, 0] = 7
I, J, _ = sparse.find(B)
A = sparse.csc_matrix((I + 1, (I, J)))
A.toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 256. Substitution

# + pycharm={"name": "#%%\n"}
I, J, V = sparse.find(B)
A = sparse.csc_matrix((V**2, (I, J)))
A.toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 257. Substitution

# + pycharm={"name": "#%%\n"}
A = B.copy()
A[:, 1:4] = sparse.csc_matrix(([6] * len(I), (I, J)))[:, 1:4]
A.toarray()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 258. Conversion from sparse to full

# + pycharm={"name": "#%%\n"}
A = sparse.rand(8, 8, 0.3)
B = A.todense()
C = sparse.csc_matrix(B)
np.max(np.absolute(A - C))

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 262. Typical operations: Delete small entries

# + pycharm={"name": "#%%\n"}
A = sparse.random(10, 10, 0.3)
I, J, V = sparse.find(A)
K = np.absolute(V) > 0.2
sparse.csc_matrix((V[K], (I[K], J[K])), (10, 10)).toarray()
# -


