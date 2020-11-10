# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex4
"""
import numpy as np
v = np.asarray([1, 4, 4, 3, 2, 0, 1.5, 1, 4, 3])  # reset

v[[2, 5, 7]] = [[8, 7, -7]]
print(f'a) : {v}')

v = np.asarray([1, 4, 4, 3, 2, 0, 1.5, 1, 4, 3])  # reset
v[2:6] = [8, 7, -7, 1]
print(f'b) : {v}')

v = np.asarray([1, 4, 4, 3, 2, 0, 1.5, 1, 4, 3])  # reset
v[v == 4] = -7
print(f'c) : {v}')

v = np.asarray([1, 4, 4, 3, 2, 0, 1.5, 1, 4, 3])  # reset
print(f'd) {np.logical_and(v >= 2, v < 4)}')

print(f'e) {np.logical_and(v >= 2, v < 4).nonzero()[0]}')

print(f'f) {len(v[v==4])}')

print(f'g) {np.delete(v, [2, 5, 7])}')

print(f'h) {v[v.nonzero()]}')

print(f'i) {len(v[v.nonzero()])}')
