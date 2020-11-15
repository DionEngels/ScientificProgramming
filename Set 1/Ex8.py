# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex8
"""
import numpy as np
import given_material.mlext as me
from given_material.mlext import print_matrix


def ex8a(matrix, value):
    matrix[matrix < value] = 0
    return matrix


def ex8bc(matrix, n_col_row, n_axis):
    return np.delete(matrix, n_col_row, axis=n_axis)


def ex8d(matrix, values):
    matrix[values] = 0
    return matrix


def ex8e_1(size):
    return np.diag(np.arange(1, size+1, dtype=int))


size = 8
A = me.magic(size)
print_matrix('Base matrix:', A)
print_matrix('Result a)', ex8a(A.copy(), 32))
print_matrix('Result b)', ex8bc(A.copy(), 4, 0))
print_matrix('Result c)', ex8bc(A.copy(), -1, 1))

input_I = [3, 1, 3, 6]
input_J = [1, 2, 7, 4]
print_matrix('Result d)', ex8d(A.copy(), (input_I, input_J)))
print_matrix('Result e)', ex8e_1(size))

