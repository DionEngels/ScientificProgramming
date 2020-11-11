# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex4
"""
import numpy as np


def ex4abc(array, indices, values):
    array[indices] = values
    return array


def ex4d(array, lb, ub):
    return np.logical_and(array >= lb, array < ub)


def ex4e(array, lb, ub):
    return np.logical_and(array >= lb, array < ub).nonzero()[0]


def ex4f(array, value):
    return len(array[array == value])


def ex4g(array, to_delete):
    return np.delete(array, to_delete)


def ex4h(array):
    return array[array.nonzero()]


def ex4i(array):
    return len(array[array.nonzero()])


v = np.asarray([1, 4, 4, 3, 2, 0, 1.5, 1, 4, 3])  # reset
print(f'Base array \t: {v}')
print(f'a) Result \t: {ex4abc(v.copy(), [2, 5, 7], [8, 7, -7])}')
print(f'b) Result \t: {ex4abc(v.copy(), slice(2,6), [8, 7, -7, 1])}')
print(f'c) Result \t: {ex4abc(v.copy(), v==4, -7)}')
print(f'd) Result \t: {ex4d(v.copy(), 2, 4)}')
print(f'e) Result \t: {ex4e(v.copy(), 2, 4)}')
print(f'f) Result \t: {ex4f(v.copy(), 4)}')
print(f'g) Result \t: {ex4g(v.copy(), [2, 5, 7])}')
print(f'h) Result \t: {ex4h(v.copy())}')
print(f'i) Result \t: {ex4i(v.copy())}')
