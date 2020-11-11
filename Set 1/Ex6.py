# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex6
"""
import numpy as np


def ex6a(array):
    return np.argmin(array)


def ex6b(array):
    return np.where(array == np.min(array))[0]


def ex6c(array, max_val):
    return [len(array[array == value]) for value in range(1, max_val)]


k = 9
v = np.random.randint(low=1, high=k, size=10**7)

print(f'a) Result \t: {ex6a(v)}')
print(f'b) Result \t: {ex6b(v)}')
print(f'c) Result \t: {ex6c(v, k)}')
