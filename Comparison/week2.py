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

# # Matrices
#
# ## 143. Example matrix creation

# + pycharm={"name": "#%%\n"}
M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
M
# -

# A numpy array/matrix is created by passing in a python array/matrix.
# According to the numpy documentation, don't use the `np.matrix()`, but use regular numpy arrays to create a matrix:
#
# > It is no longer recommended to use this class, even for linear algebra. Instead use regular arrays. The class may be removed in the future.
#
# When passing in an array, both `(...)` and `[...]` are valid syntax.

# + pycharm={"name": "#%%\n"}
import numpy as np
np.array(M)


# + pycharm={"name": "#%%\n"}
np.array(((1, 2, 3), (4, 5, 6)))
# -

# To create a matrix from columns, use one of the following:

# + pycharm={"name": "#%%\n"}
columns = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]).transpose()

# + pycharm={"name": "#%%\n"}
np.column_stack([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
# -

# ## 145. Example matrix composition
#
# Append columns: see above.
# Append rows:

# + pycharm={"name": "#%%\n"}
np.stack(([1, 2, 4], [5, 4, 3]))
# -

# To actually concatenate matrices, use `np.concatenate()` with the appropriate axis (0 for rows, 1 for columns).

# + pycharm={"name": "#%%\n"}
A = np.array(M)
B = np.array(M).transpose()

np.concatenate((A, B), axis=0)  # concatenate rows (default)

# + pycharm={"name": "#%%\n"}
np.concatenate((A, B), axis=1)  # concatenate columns
# -

# ## 148. Diagonal part remark
#
# `np.diag(A)` behaves similar to MATLABs `diag(A)`:

# + pycharm={"name": "#%%\n"}
A = np.random.random((3,3))
A

# + pycharm={"name": "#%%\n"}
np.diag(A)
# -

# ## 160. Layout
#
# Interesting read in the numpy documentation about [multidimensional array indexing order issues](https://numpy.org/doc/stable/reference/internals.html#multidimensional-array-indexing-order-issues).
# The takeaway from this is that numpy offers both a row- and column-major order (row-major order denoted by C, column-major order denoted by F (for FORTRAN)).
# The default is to use row-major order, watch out when going for a column-major approach (see the last paragraph).
#
# Note: MATLAB uses column-major order (like FORTRAN). Thus the complexity table refers to the Python numpy column-major order.

# + pycharm={"name": "#%%\n"}
A = np.array(M, order='C')

# + pycharm={"name": "#%%\n"}
B = np.array(M, order='F')

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 166. Complexity
#
# According to [this answer](https://stackoverflow.com/a/52208910/6629569) on SO (stack overflow) to the question if there exists a table of time complexity of numpy operations (as we have on the slides):
#
# > BigO complexity is not often used with Python and numpy.
# > It's a measure of how the code scales with problem size.
# > That's useful in a compiled language like C.
# > But here the code is a mix of interpreted Python and compiled code.
# > Both can have the same bigO, but the interpreted version will be orders of magnitude slower.
# > That's why most of the SO questions about improving numpy speed, talk about 'removing loops' and 'vectorizing'.
#
# I suppose the table will be similar, but my guess is that it might not make much sense, other than that (obviously) operations on matrices are slower than operations on vectors.
# There seems to be no significant difference between speed in column selection and row selection:

# + pycharm={"name": "#%%\n"}
from timeit import repeat

for i in [1000, 10000]:
    n = int(i)
    print(min(repeat(
        "for j in range(len(A)): A[:,j]",
        setup=f"import numpy as np; A = np.random.randint(1, 9, ({n}, {n}))",
        repeat=3,
        number=100
    )))

# + pycharm={"name": "#%%\n"}
for i in [1000, 10000]:
    n = int(i)
    print(min(repeat(
        "for j in range(len(A)): A[j,:]",
        setup=f"import numpy as np; A = np.random.randint(1, 9, ({n}, {n}))",
        repeat=3,
        number=100
    )))

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 159. Random numbers

# + pycharm={"name": "#%%\n"}
np.array([])

# + pycharm={"name": "#%%\n"}
np.random.randint(1, 8, (8,8))

# + pycharm={"name": "#%%\n"}
np.random.rand(8*8).reshape((8,8))

# + pycharm={"name": "#%%\n"}
np.random.random((8,8))

# + pycharm={"name": "#%%\n"}
np.random.poisson(0.7, (8,8))

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 171. Visualization

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt

n = 10
A = np.random.random((n,n))

plt.matshow(A)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 172. Comparison (example)

# + pycharm={"name": "#%%\n"}
A = np.zeros((3,3))
B = np.ones((3,3))
np.linalg.norm(A - B, np.inf)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 173-174. Read/write (elements and columns)

# + pycharm={"name": "#%%\n"}
A = np.random.randint(0, 9, (3,3))
A

# + [markdown] pycharm={"name": "#%% md\n"}
# Read entry

# + pycharm={"name": "#%%\n"}
A[1, 2]

# + [markdown] pycharm={"name": "#%% md\n"}
# Write entry

# + pycharm={"name": "#%%\n"}
A[1, 2] = 0
A

# + [markdown] pycharm={"name": "#%% md\n"}
# Read/write column

# + pycharm={"name": "#%%\n"}
A[:, 1] = [0, 1, 2]
A

# + [markdown] pycharm={"name": "#%% md\n"}
# Read/write row

# + pycharm={"name": "#%%\n"}
A[1, :] = [11, 12, 13]
A

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 175-176. Entry read/write blocks

# + pycharm={"name": "#%%\n"}
A[[1,2]]

# + pycharm={"name": "#%%\n"}
A[[1,2], :]

# + [markdown] pycharm={"name": "#%% md\n"}
# To obtain a submatrix, use slicing:

# + pycharm={"name": "#%%\n"}
A[1:, 1:]

# + [markdown] pycharm={"name": "#%% md\n"}
# Note that the right index in a slice is exclusive:

# + pycharm={"name": "#%%\n"}
B = np.random.randint(1, 9, (4,4))
B

# + pycharm={"name": "#%%\n"}
B[1:3, 1:3]

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 177. Entry read/write: Efficiency
#
# As mentioned above: no significant difference between row and column selection.
# This claim might need some further research.
# This also goes for the subsequent slides.
# My current guess is that it does not make much of a difference, but maybe it does when doing some heavy work?

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 183. Logicals/Filters

# + pycharm={"name": "#%%\n"}
A

# + pycharm={"name": "#%%\n"}
A[A > 4]

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 184. Unique/Sorting

# + pycharm={"name": "#%%\n"}
I = np.random.randint(1, 20, (20, 20))
I

# + [markdown] pycharm={"name": "#%% md\n"}
# `np.sort()` sorts each column (independently).

# + pycharm={"name": "#%%\n"}
np.sort(I)

# + [markdown] pycharm={"name": "#%% md\n"}
# `np.unique()` lists the unique elements in a (multidimensional) array.

# + pycharm={"name": "#%%\n"}
np.unique(I)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 186. Arithmetic

# + pycharm={"name": "#%%\n"}
A = np.random.randint(1, 20, (4,4))
c = 3
A

# + pycharm={"name": "#%%\n"}
c * A

# + pycharm={"name": "#%%\n"}
A * c

# + [markdown] pycharm={"name": "#%% md\n"}
# Hadamard multiplication

# + pycharm={"name": "#%%\n"}
A * A

# + [markdown] pycharm={"name": "#%% md\n"}
# Matrix multiplication

# + pycharm={"name": "#%%\n"}
A.dot(B)

# + pycharm={"name": "#%%\n"}
np.dot(A, A)

# + pycharm={"name": "#%%\n"}
A @ A

# + pycharm={"name": "#%%\n"}
A.__matmul__(A)

# + [markdown] pycharm={"name": "#%% md\n"}
# Hadamard power

# + pycharm={"name": "#%%\n"}
A ** 2

# + pycharm={"name": "#%%\n"}
np.linalg.matrix_power(A, 2)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 187. Arithmetic

# + pycharm={"name": "#%%\n"}
A = np.random.randint(1, 4, (4,4))
b = np.random.randint(1, 4, (4, 1))
print(A)
print(b)
# -

# Left division ($$Ax = b$$)

# + pycharm={"name": "#%%\n"}
np.linalg.solve(A, b)

# + [markdown] pycharm={"name": "#%% md\n"}
# Left division by solving the least squares problem to minimize $$||Ax - b||_2$$
# also returns solution for singular $A$.

# + pycharm={"name": "#%%\n"}
# rcond=None to use future default for precision
np.linalg.lstsq(A, b, rcond=None)[0]
# -

# Numpy/Python does not have a right division like MATLAB does, instead, transpose both arguments and use left division:

# + pycharm={"name": "#%%\n"}
B = np.random.randint(1, 4, (4, 4))
# rcond=None to use future default for precision
np.linalg.lstsq(B.T, A.T, rcond=None)[0].T

# + [markdown] pycharm={"name": "#%% md\n"}
# # 189.  Tensors
# Tensors in numpy are simply multi-dimensional arrays.

# + pycharm={"name": "#%%\n"}
T = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])
T

# + pycharm={"name": "#%%\n"}
T[1, 0, 1]

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 193. Visualization
# Visualization uses lots of memory. If you use PyCharm and are limited to 4GB you may not have enough memory ...
# To prevent this notebook from getting very slow, the visualization part is in the separate notebook: `week2_visualization`.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 198. Block matrices
# numpy has block matrix support, for example

# + pycharm={"name": "#%%\n"}
np.block([
    [np.eye(3) * 3, np.zeros((3, 2))],
    [np.zeros((2, 3)), np.eye(2) * 2]
])

# + pycharm={"name": "#%%\n"}
plt.spy(np.random.randint(0, 5, (100, 100)))

# -


