# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex2
"""
import numpy as np


def ex2a_single(max_value, array):
    return [list(array).count(value) for value in range(1, max_value)]


def ex2a_self(max_value, array):
    result = np.zeros(max_value, dtype=int)
    for value in array:
        result[value] += 1
    return result[1:]


k = 10
n = 7

v = np.random.randint(low=1, high=k, size=n)
res_2a_single = ex2a_single(k, v)
res_2a_self = ex2a_self(k, v)
print('Result 2a single line: {}'.format(res_2a_single))
print('Result 2a self: {}'.format(res_2a_self))


def ex2b_single(max_value, array):
    return [list(array).index(value) if value in array else -1 for value in range(1, max_value)]


res_2b_single = ex2b_single(k, v)
print('Result 2b: {}'.format(res_2a_single))
