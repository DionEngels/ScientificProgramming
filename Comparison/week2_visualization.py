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

# + pycharm={"name": "#%%\n"}
import numpy as np

from matplotlib.pyplot import imshow
from PIL import Image
image = Image.open("../images/port-castellammare-del-golfo.jpg")
image

# + pycharm={"name": "#%%\n"}
im_array = np.asarray(image)
R = np.array(im_array[:, :, 0])
G = np.array(im_array[:, :, 1])
B = np.array(im_array[:, :, 2])
S = (R > 150) & (G > 150)
R[S] = 0
G[S] = 255
imshow(np.stack((R, G, B), axis=2))
# -


