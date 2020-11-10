# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex3
"""
import numpy as np
v = np.zeros(6, dtype=np.uint32)

v[[0, 2, 4, 2, 0]] = v[[0, 2, 4, 2, 0]] + 1
print(f'Result given: {v}')
v_manual = np.asarray([2, 0, 1, 0, 2, 0])
print(f'Possible solution: {v_manual}')
