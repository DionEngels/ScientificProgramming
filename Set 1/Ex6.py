# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex6
"""
import numpy as np
from Ex2 import ex2a


def ex6a(array):
    return np.argmin(array)


def ex6b(array):
    return np.where(array == np.min(array))[0]


k = 9
v = np.random.randint(low=1, high=k, size=10**7)

print(f'a) Result \t: {ex6a(v)}')
print(f'b) Result \t: {ex6b(v)}')
print(f'c) Result \t: {ex2a(k-1, v)}')
