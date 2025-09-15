import jax.numpy as jnp
import jax
from jax import Array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def overlay(base, overlay):
    return jnp.where(base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))

def mandelbrot_step(z, dz_dc, c, count, escape_radius, has_escaped):
    dz_dc = jnp.where(has_escaped, dz_dc, 2.0 * dz_dc * z + 1)
    z = jnp.where(has_escaped, z, z * z + c)

    has_escaped = jnp.abs(z) > escape_radius
    count += ~has_escaped

    return z, dz_dc, count, has_escaped

def stripe_average(z, stripe_a, has_escaped, stripe_density=2, stripe_memory=0.9):
    stripe_i = (1 + jnp.sin(stripe_density * jnp.angle(z))) / 2

    return jnp.where(has_escaped, stripe_a, stripe_memory * stripe_a + (1 - stripe_memory) * stripe_i)

def compute_lighting(
    normal_vector: Array,
    light_vector: Array,
    has_escaped: Array,
    opacity: float = .75,
    k_specular: int = 30,
    k_shininess: float = .5,
    k_diffuse: float = .5,
    k_ambient: float = .2,
) -> Array:
    """Computes the lighting for a pixel using Phong reflection model.
    See https://en.wikipedia.org/wiki/Phong_reflection_model#Applications for more details.

    Args:
        normal_vector (Array): Normal vector for all pixels (complex type)
        light_vector (Array): Light source (normalized) vector (3D)
        has_escaped (Array): If the pixel has escaped the mandelbrot set
        opacity (float, optional): Percent of received light. Defaults to .75.
        k_specular (int, optional): Specular coefficient. Defaults to 20.
        k_shininess (float, optional): Shininess coefficient. Defaults to .5.
        k_diffuse (float, optional): Diffuse coefficient. Defaults to .5.
        k_ambient (float, optional): Ambient coefficient. Defaults to .2.

    Returns:
        Array: Computed lighting for the pixel
    """
    normal_vector /= jnp.abs(normal_vector)
    normal_vector = jnp.stack([jnp.real(normal_vector), jnp.imag(normal_vector), jnp.ones_like(normal_vector, dtype=jnp.float32)], axis=-1)
    normal_vector /= jnp.linalg.norm(normal_vector, axis=2, keepdims=True)
    normal_vector *= has_escaped[..., None] # Mask pixels inside the mandelbrot set

    diffuse_light = jnp.dot(normal_vector, light_vector)

    reflection_vector = 2 * jnp.dot(normal_vector, light_vector)[..., None] * normal_vector - light_vector[None, None, :]
    viewer_vector = jnp.array([0.0, 0.0, 1.0])
    specular_light = jnp.dot(reflection_vector, viewer_vector)**k_specular

    brightness = (
        k_ambient
        + k_diffuse * diffuse_light
        + k_shininess * specular_light
    ) * opacity + (1 - opacity) / 2

    return brightness


def create_light_source(azimuth: float = 45, elevation: float = 45) -> Array:
    """Creates a light source in 3D space.

    Args:
        azimuth (float): Azimuth of the light (0° = From bottom, 90° = From right)
        elevation (float): Elevation of the light (0° = Inside mandelbrot plane, 90° = From above)

    Returns:
        Array: Light source (normalized) vector
    """
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    x = np.cos(elevation) * np.sin(azimuth)
    y = np.cos(elevation) * np.cos(azimuth)
    z = np.sin(elevation)

    return jnp.array([x, y, z])

def compute_smooth_iter(z, dz_dc, has_escaped, escape_radius=2.0):
    abs_z = jnp.abs(z)
    log_ratio = jnp.log(abs_z) / jnp.log(escape_radius)
    smooth_iter = 1 - jnp.log(log_ratio) / jnp.log(2)
    smooth_iter = jnp.where(has_escaped, smooth_iter, 0.0)

    milnor_distance = abs_z * jnp.log(abs_z) / jnp.abs(dz_dc) / 2
    milnor_distance = jnp.where(has_escaped, milnor_distance, 0)

    return smooth_iter, milnor_distance

def mandelbrot(width, height, max_iter, xlim, ylim, D=1):
    escape_radius = 10**3
    stripe_density = 2
    stripe_memory = 0.9

    xs = jnp.linspace(xlim[0], xlim[1], width * D, dtype=jnp.float32)
    ys = jnp.linspace(ylim[0], ylim[1], height * D, dtype=jnp.float32)
    X, Y = jnp.meshgrid(xs, ys)

    c = X + 1j * Y
    z = jnp.zeros_like(c, dtype=jnp.complex64)
    dz_dc = jnp.ones_like(c, dtype=jnp.complex64)
    has_escaped = jnp.zeros(c.shape, dtype=bool)
    stripe_a = jnp.zeros(c.shape, dtype=jnp.float32)

    count = jnp.zeros(c.shape, dtype=jnp.int32)

    mandelbrot_step_jit = jax.jit(mandelbrot_step)
    stripe_average_jit = jax.jit(stripe_average)
    compute_lighting_jit = jax.jit(compute_lighting)
    compute_smooth_iter_jit = jax.jit(compute_smooth_iter)

    for i in range(max_iter):
        z, dz_dc, count, has_escaped = mandelbrot_step_jit(z, dz_dc, c, count, escape_radius, has_escaped)
        stripe_a = stripe_average_jit(z, stripe_a, has_escaped, stripe_density, stripe_memory)

    light_source = create_light_source(45, 45)
    normal = z / dz_dc
    brightness = compute_lighting_jit(normal, light_source, has_escaped)

    smooth_iter, milnor_distance = compute_smooth_iter_jit(z, dz_dc, has_escaped, escape_radius)

    stripe_t = (1 + jnp.sin(stripe_density * jnp.angle(z))) / 2
    stripe_a = (stripe_a * (1 + smooth_iter * (stripe_memory - 1)) +
                            stripe_t * smooth_iter * (1 - stripe_memory))
    stripe_a = stripe_a / (1 - stripe_memory**count *
                                       (1 + smooth_iter * (stripe_memory - 1)))

    color = apply_color(count + smooth_iter, stripe_a, milnor_distance, brightness)

    return brightness, count + smooth_iter, milnor_distance, color

def apply_color(smooth_iter, stripe, milnor_distance, brightness, rgb_theta=(.11, .02, .92), ncycle=32):
    smooth_iter = jnp.sqrt(smooth_iter) % ncycle / ncycle
    milnor_distance = -jnp.log(milnor_distance) / 12
    milnor_distance = 1/(1 + jnp.exp(-10 * (milnor_distance - 0.5)))

    brightness = overlay(brightness, stripe) * (1 - milnor_distance) + milnor_distance * brightness

    color = (1 + jnp.sin(2 * jnp.pi * (smooth_iter[..., None] + jnp.array(rgb_theta)[None, None, :]))) * 0.5
    color = overlay(color, brightness[..., None])

    return color

def main():
    # jax.config.update("jax_enable_x64", True)
    x_lim = (-0.5503295086752807, -0.5503293049351449)
    y_lim = (-0.6259346555912755, -0.625934541001796)
    brightness, smooth_iter, milnor_distance, color = mandelbrot(1920, 1080, 5000, x_lim, y_lim)
    # print(color)

    plt.imshow(color)
    plt.show()


if __name__ == "__main__":
    main()
    # time_mandelbrot(1920, 1080, 500, (-2.6, 2.6), (-1.5, 1.5))
