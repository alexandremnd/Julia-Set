import numba
import numpy as np
import math

@numba.cuda.jit
def compute_julia_set(viewport, max_iterations, out):
    y, x = numba.cuda.grid(2)

    if y >= out.shape[0] or x >= out.shape[1]:
        return

    real_span = viewport[2] - viewport[0]
    imag_span = viewport[3] - viewport[1]

    z = complex(0, 0)
    c = complex(viewport[0] + x * real_span / out.shape[0], viewport[1] + y * imag_span / out.shape[1])
    out[y, x] += 1 # math.exp(-z.real ** 2 - z.imag ** 2)

    for _ in range(max_iterations):
        z = z*z + c
        modulus = z.real ** 2 + z.imag ** 2
        out[y, x] += math.exp(-modulus)

        if modulus > 4:
            return

class Mandelbrot(object):
    def __init__(self, image_size: np.ndarray, viewport: np.ndarray, max_iterations: int, oversampling: int = 4):
        """Builds a Mandelbrot object

        Args:
            image_size (float, float): Output image size
            viewport (float, float, float, float): Complex space viewport
            max_iterations (int): Maximum number of iterations for the Mandelbrot set
            oversampling (int, optional): For each pixel in the output image, a grid of
            [oversampling, oversampling] subpixels will be used to average the color. Defaults to 4.
        """
        self.image_size = image_size
        self.viewport = viewport
        self.oversampling = oversampling
        self.max_iterations = max_iterations

        self.size_os_x = self.image_size[0] * self.oversampling
        self.size_os_y = self.image_size[1] * self.oversampling
        self.image = np.zeros((self.size_os_x, self.size_os_y, 3), dtype=np.float64)

    @property
    def is_gpu_supported(self):
        return numba.cuda.is_available()

    def compute(self):
        threads_per_block = (32, 32)
        blocks_per_grid = (self.size_os_x // threads_per_block[0] + 1, self.size_os_x // threads_per_block[1] + 1)

        compute_julia_set[blocks_per_grid, threads_per_block](self.viewport, self.max_iterations, self.image)

        self.image /= np.max(self.image)
        self.image = np.reshape(self.image, (self.image_size[0], 3, self.image_size[1], 3, 3)).mean(axis=(1, 3))

        return image