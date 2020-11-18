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
# # Matrices with known eigenpairs
#
# ## 496. Block block Toeplitz matrices

# + pycharm={"name": "#%%\n"}
import scipy.sparse as sparse

n = 2
B = sparse.diags([-1, 2, -1], [-1, 0, 1], (n, n), format='csc')
I = sparse.eye(n, n)
A = sparse.kron(I, sparse.kron(I, B)) \
  + sparse.kron(I, sparse.kron(B, I)) \
  + sparse.kron(B, sparse.kron(I, I))
A.A


# + [markdown] pycharm={"name": "#%% md\n"}
# # Boundary value problems
#
# ## 509. Boundary value problems

# + pycharm={"name": "#%%\n"}
def boundary_value_problem_2(n, ax, ay, bx, by, c):
    h = 1 / (n + 1)
    x_derivative = sparse.diags([ax - h * bx / 2, -2 * ax + h**2 * c/3, ax + h * bx / 2], [-1, 0, 1], (n, n))
    y_derivative = sparse.diags([ay - h * by / 2, -2 * ay + h**2 * c/3, ay + h * by / 2], [-1, 0, 1], (n, n))
    I = sparse.eye(n, n)
    return sparse.kron(I, x_derivative) + sparse.kron(y_derivative, I)


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 510. Boundary value problems

# + pycharm={"name": "#%%\n"}
def boundary_value_problem_3(n, ax, ay, az, bx, by, bz, c):
    h = 1 / (n + 1)
    x_derivative = sparse.diags([ax - h * bx / 2, -2 * ax + h**2 * c/3, ax + h * bx / 2], [-1, 0, 1], (n, n))
    y_derivative = sparse.diags([ay - h * by / 2, -2 * ay + h**2 * c/3, ay + h * by / 2], [-1, 0, 1], (n, n))
    z_derivative = sparse.diags([az - h * by / 2, -2 * az + h**2 * c/3, az + h * bz / 2], [-1, 0, 1], (n, n))
    I = sparse.eye(n, n)
    return sparse.kron(I, sparse.kron(I, x_derivative)) \
         + sparse.kron(I, sparse.kron(y_derivative, I)) \
         + sparse.kron(z_derivative, sparse.kron(I, I))


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 511/512. BVP 2D

# + pycharm={"name": "#%%\n"}
import numpy as np
from math import pi
from scipy.sparse.linalg import spsolve

a_x, a_y = -1, -1
b_x, b_y = 4, -3
n = 100
A = boundary_value_problem_2(n, a_x, a_y, b_x, b_y, 0)

[X, Y] = np.meshgrid(np.arange(1, n+1)/(n+1), np.arange(1, n+1)/(n+1))

h = 1 / (n + 1)
# Use np.sin and np.cos as those can be applied to an iterable.
f = h**2 * np.reshape(
    (a_x * (-4 * pi**2 * np.sin(2 * pi * X)) + b_x * 2 * pi * np.cos(2 * pi * X)) * np.sin(2 * pi * Y) +
    (a_y * (-4 * pi**2 * np.sin(2 * pi * Y)) + b_y * 2 * pi * np.cos(2 * pi * Y)) * np.sin(2 * pi * X)
    , (n**2, 1)
)

u = spsolve(A, f)
u_bvp = np.reshape(np.sin(2 * pi * X) * np.sin(2 * pi * Y), (n**2, 1))

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
from matplotlib import cm  # color map

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, np.reshape(u, (n, n)), cmap=cm.jet)

fig.colorbar(surf)
ax.azim = 230 # Change the camera angle so it is the same as in the matlab example
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z");  # ; to suppress output

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 513. BVP 2D
# This is the same as the previous example, but with a different $$f$$, right?
# Then maybe one example would be enough?

# + pycharm={"name": "#%%\n"}
f = h**2 * np.reshape(
    (2 * a_x + b_x * (2 * X - 1)) * Y * (Y - 1)
    + (2 * a_y + b_y * (2 * Y - 1)) * X * (X - 1)
    , (n**2, 1)
)
u = spsolve(A, f)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, np.reshape(u, (n, n)), cmap=cm.jet)

fig.colorbar(surf)
ax.azim = 230 # Change the camera angle so it is the same as in the matlab example
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z");

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 514. BVP 3D

# + pycharm={"name": "#%%\n"}
n = 10
a_x, a_y, a_z = -1, -1, -1
b_x, b_y, b_z = -3, 4, 0
c = 0

A = boundary_value_problem_3(n, a_x, a_y, a_z, b_x, b_y, b_z, c)

[X, Y, Z] = np.meshgrid(np.arange(1, n+1)/(n+1), np.arange(1, n+1)/(n+1), np.arange(1, n+1)/(n+1), indexing='ij')

f = h**2 * np.reshape(
    np.sin(4 * pi * (X**2)) + Y**2 + Z**2
    , (n**3, 1)
)

u = spsolve(A, f)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 520. Triangular/diagonal parts

# + pycharm={"name": "#%%\n"}
A = sparse.random(5, 5, 0.75, data_rvs=np.ones)
sparse.triu(A, 0).A

# + pycharm={"name": "#%%\n"}
sparse.tril(A, 0).A

# + pycharm={"name": "#%%\n"}
A.diagonal()

# + pycharm={"name": "#%%\n"}
sparse.diags([A.diagonal()], [0], (5, 5)).A


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 532. LDU factorization

# + pycharm={"name": "#%%\n"}
def ldu(A):
    A_ = A.copy()
    for k in range(1, len(A)):
        A_[k:, k:] = A_[k:, k:] - (1/A_[k-1, k-1]) * (np.reshape(A_[k:, k-1], (len(A_)-k, 1)) @ np.reshape(A_[k-1, k:], (1, len(A_) - k)))
    return A_

def diag_inverse(D):
    return np.diag(1 / np.diag(D))


# + pycharm={"name": "#%%\n"}
A = np.array([
    [1, 0, 1, 0],
    [2, 1, -1, 0],
    [0, -1, 0, 1],
    [-2, 0, 1, 3]
])

LDU = ldu(A)
D = np.diag(np.diag(LDU))
L = np.tril(LDU, -1)
U = np.triu(LDU, 1)
(L + D) @ diag_inverse(D) @ (D + U)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 534. Solving $$Ax = b$$ via LDU
#
# What if `U(i, i)` is 0 in the line `x(i) = x(i)/U(i, i)`?
# In this example there is no zero diagonal element, but maybe it should be mentioned that this is a condition (and that it may be assumed, in the exercise)?

# + pycharm={"name": "#%%\n"}
n = 6
U = sparse.triu(sparse.random(n, n, 1), format='csc')
b = U @ np.reshape(range(1, n+1), (n, 1))
x = b.copy()

for i in range(n)[::-1]:
    x[i] = x[i] / U[i, i]
    if i > 0:
        x[:i] = x[:i] - (x[i][0] * U[:i, i])

np.linalg.norm(U @ x - b, np.inf)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 535. Solving $$Ax =b$$ via LDU

# + pycharm={"name": "#%%\n"}
n = 6
L = sparse.tril(sparse.random(n, n, 1), format='csc')
b = L @ np.reshape(range(1, n+1), (n, 1))
x = b.copy()

for i in range(n):
    x[i] = x[i] / L[i, i]
    if i < n-1:
        x[i+1:n] = x[i+1:n] - (x[i][0] * L[i+1:n, i])

np.linalg.norm(L @ x - b, np.inf)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 537. Factorizations and round-off errors

# + pycharm={"name": "#%%\n"}
from scipy import linalg as la
import scipy.io

mat = scipy.io.loadmat('../data/ST2.mat')
X = mat['X']
# Does not take sparse matrices...
[L, D, P] = la.ldl(X.toarray())
I, J = np.ix_(P, P)

plt.spy((L @ D @ L.T) - X[I, J])
np.linalg.norm(L @ D @ L.T - X[I, J], np.inf)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 542. Factorizations and fill-in

# + pycharm={"name": "#%%\n"}
A = np.zeros((6, 6))
A[0, :] = np.ones(6)
A[:, 0] = np.ones(6)
# Documentation says that the second argument should be a scalar used to fill the diagonal.
# However, at the bottom of the page are examples that show the use of an array instead: https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
# No guarantee this will work in all cases, but it does work here:
np.fill_diagonal(A, [2**i for i in range(6)])
A

# + pycharm={"name": "#%%\n"}
# As defined somewhere above
ldu(A)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 545. Factors L + U

# + pycharm={"name": "#%%\n"}
from scipy import linalg as la
import scipy.io

mat = scipy.io.loadmat('../data/ST2.mat')
X = mat['X']
[P, L, U] = la.lu(X.toarray())
plt.spy(L + U)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 546. (?) Factors L + U of `[L, U] = lu(X(P, P))`
#
# Reverse Cuthill-McKee:

# + pycharm={"name": "#%%\n"}
import scipy.sparse.csgraph as csgraph

p = csgraph.reverse_cuthill_mckee(X)
I, J = np.ix_(p, p)
A = X[I, J]
[Q, L, U] = la.lu(A.toarray())
plt.spy(L + U)

# + [markdown] pycharm={"name": "#%% md\n"}
# BFS ordering:

# + pycharm={"name": "#%%\n"}
p, _ = csgraph.breadth_first_order(X, 0)
I, J = np.ix_(p, p)
A = X[I, J]
[Q, L, U] = la.lu(A.toarray())
plt.spy(L + U)
