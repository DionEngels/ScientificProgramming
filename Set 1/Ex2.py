# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex2
"""
import numpy as np
from given_material.mlext import accumarray


def ex2a(max_value, array):
    uniques = np.unique(array, return_counts=True)
    return accumarray(uniques[0] - 1, uniques[1], sz=max_value, fillval=0).astype(int)


def ex2b(max_value, array):
    uniques = np.unique(array, return_index=True)
    return accumarray(uniques[0] - 1, uniques[1] + 1, sz=max_value, fillval=0).astype(int)


k = 10
n = 7

v = np.random.randint(low=1, high=k, size=n)
print(f'Result 2a \t: {ex2a(k, v)}')
print(f'Result 2b \t: {ex2b(k, v)}')
