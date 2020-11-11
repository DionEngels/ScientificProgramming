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


def ex2b_single(max_value, array):
    return [list(array).index(value) if value in array else -1 for value in range(1, max_value)]


k = 10
n = 7

v = np.random.randint(low=1, high=k, size=n)
print(f'Result 2a single line \t: {ex2a_single(k, v)}')
print(f'Result 2a self \t\t\t: {ex2a_self(k, v)}')
print(f'Result 2b \t\t\t\t: {ex2b_single(k, v)}')
