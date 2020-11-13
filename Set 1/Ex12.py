# -*- coding: utf-8 -*-
"""
Created on 13/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex12
"""

import timeit
import numpy as np
import functools
from scipy import sparse


def f(n: int) -> np.array:
    A = np.zeros((n, n))
    for i in range(0, n):
        A[i, i] = 2
        if i > 1:
           A[i, i - 1] = -1
        if i < n-1:
            A[i, i + 1] = -1
    return A


def tridiag(n: int) -> np.array:
    return sparse.diags([-1, 2, -1], offsets=[-1, 0, 1], shape=(n,n))


def tridiag2(n: int) -> np.array:
    return sparse.diags([np.ones((n - 1)) * -1, np.ones(n) * 2, np.ones(n - 1) * -1], [-1, 0, 1])


instances = 100
n = 10000
t = timeit.Timer(functools.partial(f, n))
res_f = t.timeit(instances) / instances
t = timeit.Timer(functools.partial(tridiag, n))
res_tridiag = t.timeit(instances) / instances
t = timeit.Timer(functools.partial(tridiag2, n))
res_tridiag2 = t.timeit(instances) / instances
print(f"Timing for f: {res_f}")
print(f"Timing for tridiag: {res_tridiag}")
print(f"Timing for tridiag_alt: {res_tridiag} ", )
print(f'Difference between f and tridiag: {res_f/res_tridiag}x')