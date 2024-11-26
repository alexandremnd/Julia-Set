import math

import numpy as np
import numba
from numba import cuda

__all__ = ["julia_compute", "mandelbrot_compute", "buddhabrot_compute"]

@cuda.jit(device=True)
def square_add(z, cx, cy):
    return (z[0]**2 - z[1]**2 + cx, 2 * z[0] * z[1] + cy)

@cuda.jit(device=True)
def abs2(z):
    return z[0]**2 + z[1]**2

@cuda.jit(device=True)
def screen_space_to_world_space(x, y, zoom, size_x, size_y, origin_x, origin_y):
    # Pixel mapping: 0 <= 2x / (size_x - 1) <= 2 => -1 <= (2 * x / (size_x - 1) - 1) <= 1
    return ((2 * x / (size_x - 1) - 1) / zoom + origin_x, ((2 * y / (size_y - 1) - 1) / zoom + origin_y) * size_y / size_x)

@cuda.jit('int32(float32, float32, float32, int32, int32, float32, float32)', device=True)
def world_space_to_screen_space(x, y, zoom, size_x, size_y, origin_x, origin_y):
    screen_x = numba.int32(math.floor(0.5 * (size_x - 1) * ((x - origin_x) * zoom + 1)))
    screen_y = numba.int32(math.floor(0.5 * (size_y - 1) * ((y * size_x / size_y - origin_y) * zoom + 1)))
    
    return (screen_y, screen_x)

@cuda.jit("void(float32, float32, float32, int32, float32, float32, int32, int32, float32[:,:])")
def julia_compute(origin_x, origin_y, zoom, num_iters, cx, cy, size_x, size_y, out):
    y, x = numba.cuda.grid(2)
    
    if y >= out.shape[0] or x >= out.shape[1]:
        return
    
    z = screen_space_to_world_space(x, y, zoom, size_x, size_y, origin_x, origin_y)
    out[y, x] += math.exp(-abs2(z))
    
    for _ in range(num_iters):
        z = square_add(z, cx, cy)
        modulus = abs2(z)
        out[y, x] += math.exp(-modulus)
        
        if modulus > 4:
            return
        
@cuda.jit("void(float32, float32, float32, int32, float32, float32, float32[:,:])")
def mandelbrot_compute(origin_x, origin_y, zoom, num_iters, size_x, size_y, out):
    y, x = numba.cuda.grid(2)
    
    if y >= out.shape[0] or x >= out.shape[1]:
        return
    
    z = (0, 0)
    (cx, cy) = screen_space_to_world_space(x, y, zoom, size_x, size_y, origin_x, origin_y)
    
    for n in range(num_iters):
        z = square_add(z, cx, cy)
        modulus = abs2(z)
        
        if modulus > 4:
            out[y, x] = n + 1
            return
        
    out[y, x] = num_iters + 1
    
@cuda.jit("void(float32, float32, float32, int32, float32, float32, float32[:,:])")
def buddhabrot_compute(origin_x, origin_y, zoom, num_iters, size_x, size_y, out):
    y, x = numba.cuda.grid(2)
    
    if y >= out.shape[0] or x >= out.shape[1]:
        return
    
    z = (0, 0)
    (cx, cy) = screen_space_to_world_space(x, y, zoom, size_x, size_y, origin_x, origin_y)
    
    for _ in range(num_iters):
        z = square_add(z, cx, cy)
        if abs2(z) >= 4:
            break
        
    if abs2(z) <= 4:
        return
        
    z = (0, 0)
    (cx, cy) = screen_space_to_world_space(x, y, zoom, size_x, size_y, origin_x, origin_y)
        
    for _ in range(num_iters):
        z = square_add(z, cx, cy)
        (screen_y, screen_x) = world_space_to_screen_space(z[0], z[1], zoom, size_x, size_y, origin_x, origin_y)
        
        if (0 <= screen_x < size_x and 0 <= screen_y < size_y):
            cuda.atomic.add(out, (screen_y, screen_x), 1)
        