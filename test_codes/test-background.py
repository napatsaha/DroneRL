from utils.plot_utils import draw_background
import matplotlib.pyplot as plt
import numpy as np
import string

# Metadata
parent_dir = "test2"
run_base_name = "TestWorked"
run_ids = [1, 3, 4, 5, 8]
max_reps = 10
metric = None
save = False
show = True

# plot_info_background(
#     parent_dir,
#     run_base_name,
#     run_ids
# )

canvas = draw_background(parent_dir, run_base_name, run_ids[-1])
cmap = plt.get_cmap("Set1")

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(canvas, cmap=plt.get_cmap("gray"), extent=(0, 1, 0, 1), origin="lower")


num_rects = 11
for i in range(num_rects):
    xy = np.random.rand(2) * 0.8
    width = 0.2 # np.random.rand() * 0.4 + 0.1
    length = 0.2 # np.random.rand() * 0.4 + 0.1
    rect = plt.Rectangle(xy, width, length, ec=cmap(i), fc=cmap(i, 0.1))
    ax.add_patch(rect)
    ax.annotate(f"{string.ascii_uppercase[i]}", xy + 0.5 * np.array([width, length]),
                xytext=xy + np.array([0.75*width, 1.2 * length]),
                ha="left", color=cmap(i), fontsize=20,
                arrowprops=dict(color="grey", connectionstyle="arc3,rad=-0.25", arrowstyle= "fancy,tail_width=0.1,head_width=0.4"))

# plt.axis("off")
plt.show()
