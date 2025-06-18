#!/usr/bin/env python3

import sys
sys.path.append("/Users/giroux/GitHub/pymmr")

import numpy as np
import matplotlib.pyplot as plt

from pymmr.mmr import normal_field

c1c2 = np.array(
    [[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0]]
)

xo = np.linspace(-800, 800, 52)
yo = np.linspace(-400, 400, 26)
xo = np.c_[
    np.kron(xo.reshape(-1, 1), np.ones((yo.size, 1))),
    np.kron(np.ones((xo.size, 1)), yo.reshape(-1, 1)),
    np.zeros((xo.size * yo.size, 1)),
]

B = normal_field(c1c2, xo)

Bx = B[0].reshape((52, 26)).T
By = B[1].reshape((52, 26)).T

plt.subplot(2, 1, 1)
plt.imshow(Bx)
plt.subplot(2, 1, 2)
plt.imshow(By)
plt.show()
