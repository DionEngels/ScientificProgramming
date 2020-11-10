# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex8
"""
import numpy as np
import given_material.mlext as me

A = me.magic(8)
print(f'Base matrix: {A}')

A_a = A.copy()
A_a[A_a < 32] = 0
print(f'a) {A_a}')

print(f'b) {np.delete(A, 4, axis=0) }')

print(f'c) {np.delete(A, -1, axis=1)}')

input_I = [3, 1, 3, 6]
input_J = [1, 2, 7, 4]
A_d = A.copy()

A_d[(input_I, input_J)] = 0
print(f'd) {A_d}')

## E to do
