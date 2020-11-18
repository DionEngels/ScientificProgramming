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
# # Tables and databases
#
# ## 577/578. Representation 3
#
# Given a matrix of tuples (representation 2):

# + pycharm={"name": "#%%\n"}
import numpy as np

H = np.array([
    [7, 4],
    [4, 1],
    [5, 2],
    [3, 2],
    [9, 4],
    [6, 3],
    [7, 1],
    [7, 4]
])

# + pycharm={"name": "#%%\n"}
import scipy.sparse as sparse
sparse.csc_matrix(([1] * len(H), (H[:,0] - 1, H[:, 1] - 1)), shape=(13, 4)).A

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 585. Table
# Table in matlab can often be replaced by a [pandas](https://pandas.pydata.org/) dataframe in Python.
#
# ## 586. Relabel
# Note: I could not find the real database file in `2mmn20_1.zip`, so copied the first 16 entries from the slides.
# I suppose it's good enough for the examples, but they can easily be extended with the complete database.

# + pycharm={"name": "#%%\n"}
import pandas as pd

df = pd.read_csv("../data/customer_transactions.csv", sep=" ")
df

# + pycharm={"name": "#%%\n"}
customers = df['Customer'].unique()
replacements = dict(zip(customers, range(1, len(customers) + 1)))
df['Customer indices'] = [replacements[i] for i in df['Customer']]
df


# + [markdown] pycharm={"name": "#%% md\n"}
# To replace the customer column:

# + pycharm={"name": "#%%\n"}
df['Customer'] = [replacements[i] for i in df['Customer']]
df

# + pycharm={"name": "#%%\n"}
df = pd.read_csv("../data/customer_transactions.csv", sep=" ")
distinct_timestamps = df.Timestamp.unique()
len(distinct_timestamps)

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
distinct_timestamps.sort()
plt.plot(distinct_timestamps)
plt.xlabel("distinct times")
plt.ylabel("time t")

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 590. Field analysis: timestamps

# + pycharm={"name": "#%%\n"}
np.unique(np.diff(distinct_timestamps))

# + pycharm={"name": "#%%\n"}
import math
seconds_per_day = 60 * 60 * 24

# Compute the day of the transaction
df['Day'] = [math.floor(t / seconds_per_day) for t in df['Timestamp']]

# Compute the indices of the customers
customers = df['Customer'].unique()
replacements = dict(zip(customers, range(1, len(customers) + 1)))
df['Customer indices'] = [replacements[i] for i in df['Customer']]
df

# + pycharm={"name": "#%%\n"}
time_series = sparse.csc_matrix((df['Volume'], (df['Day'] - 1, df['Customer indices'] - 1)))
plt.spy(time_series)
plt.xlabel("customer")
plt.ylabel("day");

# + pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
from matplotlib import cm  # color map

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(1, df['Customer indices'].max() + 1), range(1, df['Day'].max() + 1))
surf = ax.plot_surface(X, Y, time_series.A)
ax.set_xlabel("customer")
ax.set_ylabel("day")
ax.set_zlabel("volume");

# + [markdown] pycharm={"name": "#%% md\n"}
# ## 598. Field operations

# + pycharm={"name": "#%%\n"}
df[['Timestamp', 'Customer']].groupby(['Customer']).max()

