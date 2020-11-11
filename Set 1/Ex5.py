# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex5
"""
import numpy as np


def ex5a_1(large, small):
    return [value for value in large if value not in small]


def ex5a_opt(large, small):
    for value in small:
        large = np.delete(large, np.where(large == value))
    return large


input_I = np.asarray([8, 6, 3, 6, 8, 9])
input_J = [3, 8, 7]

print(f'a) one-line \t: {ex5a_1(input_I, input_J)}')
print(f'a) optimal \t\t: {ex5a_opt(input_I, input_J)}')
print(f'b) one-line \t: {ex5a_1(input_I, input_J)}')