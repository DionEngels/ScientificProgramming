# -*- coding: utf-8 -*-
"""
Created on 10/11/2020

----------------------------

Author: s150127
Project: ScientificProgramming
File: Ex7
"""
import numpy as np
from given_material.mlext import print_matrix


def ex7(size):
    return np.reshape(np.tile(np.array(range(1, size + 1)), size), (size, size))


n = 8
print_matrix('Result Ex7:', ex7(n))
