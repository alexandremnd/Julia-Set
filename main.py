import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import mandelbrot

def get_image_size(viewport, size_x):
    """Give required image size to keep an aspect ratio of 1:1 in the final image

    Args:
        viewport (float, float, float, float): Lower left and upper right corners of the complex plane
        size_x (int): Length of the image in the x-axis

    Returns:
        (int, int): Image size
    """
    real_span = viewport[2] - viewport[0]
    imag_span = viewport[3] - viewport[1]
    size_y = int(size_x * imag_span / real_span)
    return (size_y, size_x)


image_size = (1080, 1920)
viewport = (-2.5, -1.5, 1, 1.5)
image_size = get_image_size(viewport, image_size[1])

manderbrot_set = mandelbrot.Mandelbrot(image_size, viewport, 100, 1)
image = manderbrot_set.compute()

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