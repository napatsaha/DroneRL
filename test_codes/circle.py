# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:53:07 2023

@author: napat
"""

import numpy as np
import matplotlib.pyplot as plt

theta = 20
radius = 1
angle = np.deg2rad(np.arange(0, 360, theta))
y = radius * np.sin(angle) + radius
x = radius * np.cos(angle) + radius

plt.scatter(x,y)
# np.deg2rad(90)
