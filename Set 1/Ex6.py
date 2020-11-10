# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex6
"""
import numpy as np
k = 9
v = np.random.randint(low=1, high=k, size=10**7)

print(f'a) {np.argmin(v)}')

res_b = np.where(v == np.min(v))[0]
print(f'b) {res_b}')

print(f'c) {[len(v[v==value]) for value in range(1, k)]}')
