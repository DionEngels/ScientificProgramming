# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex7
"""
import numpy as np
n = 8

res = np.reshape(np.tile(np.array(range(1, n + 1)), n), (n, n))
