# -*- coding: utf-8 -*-
"""
Created on 11/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex9
"""
import numpy as np
from scipy import sparse
from given_material.mlext import print_matrix

np.random.seed(623827)
n = 8
d = 0.1
A = sparse.rand(n, n, d, format='csc')
print_matrix('Random matrix:', A.todense())

# create indices -- but not correct because of potential double (I,J)
N = 14
I = np.random.randint(0, n, size=N)
J = np.random.randint(0, n, size=N)

# create values
V = np.random.random(N)

# not correct for (N = 2 non-zero entries) I = np.array([0,0]); J = np.array([0,0]); V = np.array([0.8,0.8])
A = sparse.coo_matrix((V, (I, J)), shape=(n, n))
print_matrix('I/J matrix:', A.todense())


def ex9a(rows, columns, size):
    values = [10 * row + column + 1 for row, column in zip(rows, columns)]
    return sparse.coo_matrix((values, (rows, columns)), shape=(size, size))


def ex9b(matrix, value):
    matrix = matrix.todense()
    matrix[matrix == 0] = value
    return matrix


def ex9c(rows, columns, size):
    values = [column + 1 for column in columns]
    return sparse.coo_matrix((values, (rows, columns)), shape=(size, size))


def ex9d(values, rows, columns, size):
    indices = [np.where(columns == column)[0] for column in range(size)]
    return [rows[indice[np.argmax(values[indice])]] + 1 if len(indice) > 0 else 0 for indice in indices]


print_matrix('Exercise 9a)', ex9a(I.copy(), J.copy(), n).todense())
print_matrix('Exercise 9b)', ex9b(A.copy(), 2))
print_matrix('Exercise 9c)', ex9c(I.copy(), J.copy(), n).todense())
print(f'Exercise 9d) {ex9d(V.copy(), I, J, n)}')
