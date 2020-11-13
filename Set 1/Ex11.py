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
    booleans = [True if value > 0 and column == row or value < 0 and column != row else False
                for value, column, row in zip(A.data, A.col, A.row)]
    return all(booleans)


# generate matrix
np.random.seed(623827)
n = 8
d = 1
A = sparse.rand(n, n, d, format='coo')
A.data = (A.data - 0.5) * 100
A = A.astype(np.int)
ex11(A)
