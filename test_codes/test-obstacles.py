# import pygame
#
# pygame.Rect.colliderect()
#
# rect1 = pygame.Rect(0,0,25,25)
# rect2 = pygame.Rect(10,10,25,25)
#
# rect1.colliderect(rect2)

from skimage.draw import line_aa
import numpy as np
import matplotlib.pyplot as plt
import cv2

HEIGHT = 800
WIDTH = 800

canvas = np.ones((HEIGHT, WIDTH))
obs1 = line_aa(500,200,600,600)
# obstacle = line_aa(np.random.randint(WIDTH),
#                    np.random.randint(WIDTH),
#                    np.random.randint(WIDTH),
#                    np.random.randint(WIDTH))

canvas[obs1[0], obs1[1]] = obs1[2]
# plt.imshow(canvas)
# plt.show()

cv2.imshow("x", canvas)
cv2.waitKey(0)
cv2.destroyWindow("x")