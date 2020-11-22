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


def ex10_v2(N):
    ones = (N == (np.max(N[N[:, 1] == 1, 0]), 1)).all(axis=1)
    twos = (N == (np.max(N[N[:, 1] == 2, 0]), 2)).all(axis=1)
    threes = (N == (np.max(N[N[:, 1] == 3, 0]), 3)).all(axis=1)
    fours = (N == (np.max(N[N[:, 1] == 4, 0]), 4)).all(axis=1)
    return N[ones | twos | threes | fours, :]

# define
n = 14
ranks = np.random.randint(low=1, high=14, size=n)
suites = np.random.randint(low=1, high=5, size=n)
N = np.transpose(np.vstack((ranks, suites)))
print(ex10_v2(N))

