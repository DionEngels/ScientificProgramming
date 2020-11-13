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


def ex10_long(N):
    res = []
    for i in range(1, 5):
        counts = accumarray(N[N[:,1]==i, :][:,0], 1)
        if len(counts) > 0:
            res.append(np.asarray([[np.argmax(counts), i],]*np.max(counts)))
    return np.vstack(tuple(res))


# define
n = 100
ranks = np.random.randint(low=1, high=14, size=n)
suites = np.random.randint(low=1, high=5, size=n)
N = np.transpose(np.vstack((ranks, suites)))
print(ex10_long(N))

