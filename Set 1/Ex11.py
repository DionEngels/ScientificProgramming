# -*- coding: utf-8 -*-
"""
Created on 13/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex11
"""
import numpy as np
from scipy import sparse
from given_material.mlext import print_matrix


def ex11(matrix):
    matrix = matrix.tocoo()
    booleans = [True if value > 0 and column == row or value < 0 and column != row else False
                for value, column, row in zip(matrix.data, matrix.col, matrix.row)]
    return all(booleans)


# generate matrix
np.random.seed(623827)
n = 8
d = 1
A = sparse.rand(n, n, d, format='coo')
A.data = (A.data - 0.5) * 100
A = A.astype(np.int)
print(f'Result self-made: {ex11(A)}')

#%%
A = sparse.csc_matrix([
[1, 0, -2],
[0, 4, 0],
[-9, -2, 6]
])

print(f'Result B: {ex11(A)}')

#%%
A = sparse.csc_matrix([
[-2, -6, 0, 1],
[5, 11, -8, -9],
[0, 20, 22, -4],
[3, 2, 1, -7],
])

print(f'Result C: {ex11(A)}')