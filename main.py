import numba
import numpy as np
import math
from numba import cuda
import time

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from cuda_kernel import *

color_map_dict = {'red':  [(0.0,   0,  0),
                   (0.16, 32 / 255, 32 / 255),
                   (0.42,  237 / 255, 237 / 255),
                   (0.6425, 1, 1),
                   (0.8575,   0,  0),
                   (1,   0,  0)],
         'green': [(0.0, 7 / 255, 7 / 255),
                   (0.16, 107 / 255, 107 / 255),
                   (0.42, 1, 1),
                   (0.6425, 170 / 255, 170 / 255),
                   (0.8575, 2 / 255, 2 / 255),
                   (1, 2 / 255, 2 / 255)],
         'blue':  [(0.0, 100 / 255, 100 / 255),
                   (0.16, 203 / 255, 203 / 255),
                   (0.42, 1, 1),
                   (0.6425, 0, 0),
                   (0.8575, 0, 0),
                   (1, 0, 0)]}

mandelbrotCMP = LinearSegmentedColormap("mandelbrot", color_map_dict, 256)
      
(size_x, size_y) = (7680, 4320) # (7680, 4320)
image = np.zeros((size_y, size_x), dtype=np.float32)

threads_per_block = (32, 32)
blocks_per_grid = (size_y // threads_per_block[0] + 1, size_x // threads_per_block[1] + 1)

# mandelbrot_compute[blocks_per_grid, threads_per_block](0., 0., 0.5, 50, size_x, size_y, image)
stream = cuda.stream()
d_image = cuda.to_device(image, stream=stream)

buddhabrot_compute[blocks_per_grid, threads_per_block](0., 0., 0.5, 100, size_x, size_y, d_image)

d_image.copy_to_host(image, stream=stream)
stream.synchronize()

image /= np.max(image)
image = (mandelbrotCMP(image)[:, :, 0:3] * 255).astype(np.uint8)


result = Image.fromarray(image)
result.save("buddha.png", format="png")

# video = cv2.VideoWriter('video1.mp4', -1, 60, (size_x, size_y))
# cx = np.linspace(0, 1, 60 * 10)

# for c in cx:
#     print(c * 100)
#     julia_set[blocks_per_grid, threads_per_block](0, 0., 0.5, size_x, size_y, 50, c, 1 - c, image)
#     image /= np.max(image)
#     image = (mandelbrotCMP(image)[:, :, 0:3] * 255).astype(np.uint8)
#     video.write(image[:, :, ::-1])
#     image = np.zeros((size_y, size_x))
    
# cv2.destroyAllWindows()
# video.release()