"""
https://keawang.github.io/lower-the-entropy/matplotlib/notes/2020/04/01/Plotting-and-animating-colored-lines.html
"""
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import math

theta = np.linspace(0, 2*math.pi, 100)
r = np.arange(1, 11, 1)[:, np.newaxis]
x = r * np.cos(theta)
y = r * np.sin(theta)
z = r[:, 0]

fig, ax = plt.subplots()

# First we initialize our line collection and add it to our Axes
lines = []
lc = LineCollection(lines, cmap="viridis", lw=4)
ax.add_collection(lc)


def update(num):
    new_x = x[:, :num]
    new_y = y[:, :num]

    lines = [np.column_stack([xi, yi]) for xi, yi in zip(new_x, new_y)]
    # Now we update the lines in our LineCollection
    lc.set_segments(lines)
    lc.set_array(z)
    print(lc.get_segments()[0].shape)
    return lc,
ani = animation.FuncAnimation(fig, update, x.shape[1], interval=100, blit=True)

# For some reason autoscaling doesn't work here so we need to set the limits manually
ax.set(xlabel="x", ylabel="y", aspect="equal", xlim=(-12, 12), ylim=(-12, 12))
# Put this after so that the colors are scaled properly
axcb = fig.colorbar(lc)
axcb.set_label("Radius")

plt.show()