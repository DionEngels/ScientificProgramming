# -*- coding: utf-8 -*-
"""
Created on 13/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex10
"""
import numpy as np
from given_material.mlext import accumarray


def ex10(H):
    ones = (H == (np.max(H[H[:, 1] == 1, 0], initial=0), 1)).all(axis=1)
    twos = (H == (np.max(H[H[:, 1] == 2, 0], initial=0), 2)).all(axis=1)
    threes = (H == (np.max(H[H[:, 1] == 3, 0], initial=0), 3)).all(axis=1)
    fours = (H == (np.max(H[H[:, 1] == 4, 0], initial=0), 4)).all(axis=1)
    return H[ones | twos | threes | fours]

# define random if you want
n = 14
ranks = np.random.randint(low=1, high=14, size=n)
suites = np.random.randint(low=1, high=5, size=n)
H = np.transpose(np.vstack((ranks, suites)))
# define example
H = np.array([[7, 1], [3, 2], [6, 1], [4, 2], [4, 2], [7, 1], [3, 4]])
print(ex10(H))

