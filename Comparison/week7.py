# ---
# jupyter:
#   jupytext:
#     formats: src/notebooks//ipynb,src/python//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # The Matlab MEX frontend
#
# If I understood correctly, the goal of using the MEX frontend is to embed C/C++ code into matlab, as that is faster (and compiled).
# To achieve the same thing in Python, I will look into [Cython](https://cython.org/) and directly [extending Python with C++](https://docs.python.org/3/extending/extending.html).
# On first impression, the former looks easier, but the latter looks more like using the MEX frontend in Matlab.
# I will try both (this is all new to me), report my findings, and then it is up to you to decide which way to go.
#
# ## 605. Matlab native: `dotsq()`

# + pycharm={"name": "#%%\n"}
import numpy as np

def dotsq_slow1(v):
    w = np.empty((0, 0))
    for i in v:
        w = np.append(w, i**2)
    return w


# + pycharm={"name": "#%%\n"}
def dotsq_slow2(v):
    return list(map(lambda x: x**2, v))


# + pycharm={"name": "#%%\n"}
def dotsq_fast(v):
    return v**2


# + [markdown] pycharm={"name": "#%% md\n"}
# `dotsq_slow1(v)` takes around 3 seconds with vector of size $$10^5$$, and for $$10^6$$ it took a lot longer.
# Haven't tested whether it's just the `for` loop that is slow, but my guess is that a `np.append()` is also slow.
# Test this?
#
# The time of `dotsq_fast()` looks comparable to matlab.

# + pycharm={"name": "#%%\n"}
v = np.random.randint(0, 100, 100000)

# Jupyter has a built-in timeit :)
# %timeit -r 1 -n 1 dotsq_slow1(v)
# # %timeit -r 1 -n 1 dotsq_slow2(v)
# # %timeit -r 1 -n 1 dotsq_fast(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 607. Matlab native: `rollout()`

# + pycharm={"name": "#%%\n"}
def rollout_slow1(v):
    w = np.empty((0, 0))
    for i, u in enumerate(v):
        for j in range(u):
            w = np.append(w, i + 1)
    return w


# + pycharm={"name": "#%%\n"}
def rollout_slow2(v):
    w = np.empty((0, 0))
    for i, u in enumerate(v):
        w = np.append(w, np.full((1, u), i + 1))
    return w


# + pycharm={"name": "#%%\n"}
def rollout_fast(v):
    return np.repeat(range(1, len(v) + 1), v)


# + [markdown] pycharm={"name": "#%% md\n"}
# `slow1` and `slow2` are really slow (for a vector of size 'only' $$10^4$$ already).
# I suppose it's the `np.append()` that should be avoided.

# + pycharm={"name": "#%%\n"}
Cv = np.random.randint(0, 100, 1000000)

# Jupyter has a built-in timeit :)
# # %timeit -r 1 -n 1 rollout_slow1(v)
# # %timeit -r 1 -n 1 rollout_slow2(v)
# %timeit -r 1 -n 1 rollout_fast(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# # Cython
# First, an example taken from the [Cython documentation](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#fibonacci-fun).
#
# - I've installed Cython with `pip install cython`
# - Run `python setup.py build_ext --inplace` in the `fastcython` folder

# + pycharm={"name": "#%%\n"}
# Add the src directory (parent of current directory) to the Python path so it can
# find our cython modules.
import sys
sys.path.insert(1, "../")

import fastcython.fibonacci as fib
fib.fib(10)


# + [markdown] pycharm={"name": "#%% md\n"}
# To make changes in the cython file and use them here (in the notebook) I've found the following to be the shortest:
#
# - Run `python setup.py build_ext --inplace` in the `fastcython` folder
# - Restart the kernel
# - Rerun the cell (note that, for convenience, the cell includes all necesarry `import`s)
#
# Note that I have omitted the Cython code here, as it has to be compiled with C first.

# + pycharm={"name": "#%%\n"}
import numpy as np
from fastcython.dotsq import slow_dotsq_py

v = np.array([1, 2, 3, 5])
slow_dotsq_py(v)


# + [markdown] pycharm={"name": "#%% md\n"}
# ## 612. `dotsq()` in Cython (MEX: `dotsq_mex()`)
#
# To try out if simply putting a function in a Cython file and 'compiling' with Cython is enough to gain in performance, I put the `dotsq_slow1()` function (repeated below) in the Cython file as `slow_dotsq_py()`.
# The timings are below, and it looks like simply compiling with Cython is not enough to speed up.

# + pycharm={"name": "#%%\n"}
# The 'noinspection PyRedeclaration' is so that PyCharm does not complain about the redefinition of the function.
# noinspection PyRedeclaration
def dotsq_slow1(v):
    w = np.empty((0, 0))
    for i in v:
        w = np.append(w, i**2)
    return w

v = np.random.randint(0, 100, 100000)
# %timeit -r 1 -n 1 slow_dotsq_py(v)
# %timeit -r 1 -n 1 dotsq_slow1(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# However, adding explicit (C) types to all variables, and using memoryviews (see this [guide](http://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews) that I am following) speeds it up significantly.
# Not as fast as numpy, but getting closer.

# + pycharm={"name": "#%%\n"}
import numpy as np
from fastcython.dotsq import dotsq_memview as dotsq_cy

v = np.array([1, 2, 3, 5]).astype(np.intc)

dotsq_cy(v)

# + pycharm={"name": "#%%\n"}
import numpy as np
from fastcython.dotsq import dotsq_memview as dotsq_cy

# noinspection PyRedeclaration
def dotsq_fast(v):
    return v**2

v = np.random.randint(0, 100, 100000).astype(np.intc)
# %timeit -n 1000 dotsq_cy(v)
# %timeit -n 1000 dotsq_fast(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 614. `rollout()` in Cython (MEX: `rollout_mex()`)

# + pycharm={"name": "#%%\n"}
import numpy as np
from fastcython.rollout import rollout

v = np.array([1, 2, 3, 5]).astype(np.intc)

rollout(v)


# + pycharm={"name": "#%%\n"}
def rollout_fast(v):
    return np.repeat(range(1, len(v) + 1), v)

v = np.random.randint(0, 100, 1000000).astype(np.intc)
# %timeit -r 1 -n 1 rollout(v)
# %timeit -r 1 -n 1 rollout_fast(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# # C
#
# I'm following a [tutorial](https://medium.com/delta-force/extending-python-with-c-f4e9656fbf5d) that computes the $k$-th prime.
#
# - `python setup.py build_ext --inplace` to create the `.so` file (on linux)
# - Restart kernel

# + pycharm={"name": "#%%\n"}
import fastc.fastc as fastc
# %timeit fastc.kthPrime(10000)

# + pycharm={"name": "#%%\n"}
import numpy as np
import fastc.fastc as fastc
v = np.array([1, 2, 3], dtype=np.intc)
fastc.dotsq(v)

# + pycharm={"name": "#%%\n"}
import fastc.fastc as fastc
import numpy as np

# noinspection PyRedeclaration
def dotsq_fast(v):
    return v**2

v = np.random.randint(0, 100, 100000).astype(np.intc)
# %timeit -n 1000 fastc.dotsq(v)
# %timeit -n 1000 dotsq_fast(v)

# + [markdown] pycharm={"name": "#%% md\n"}
# Setting up and getting familiar with C and the Numpy/Python C-API took me around 8 hours.
# It introduces quite some boilerplate code that can be overwhelming at first.
# I've only defined the `dotsq` function, because I think the challenge here is more in the getting to know C and figuring out how to write an extension in C.
# `fastc/fastc.c` contains all this boilerplate code, and `dotsq.c` contains the 'real' function `dotsq` (forgive me if this is not C-optimal :p).
# To be able to do this well, I think students should already know C, which most students don't.
# Cython is definitely easier to pick up (and NumPy/SciPy development is also switching from C to Cython I believe).
# So my advice is to look into Cython, and 'skip' the raw C extensions.
#
