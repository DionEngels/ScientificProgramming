# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex3
"""
import numpy as np


def ex3(array, indices):
    array[indices] = array[indices] + 1
    return array


v = np.zeros(6, dtype=np.uint32)
print(f'Result given \t\t: {ex3(v, [0, 2, 4, 2, 0])}')
print(f'Possible solution \t: {np.asarray([2, 0, 1, 0, 2, 0])}')
