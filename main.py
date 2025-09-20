import jax.numpy as jnp
import jax
from jax import Array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

def overlay(base, overlay):
    return jnp.where(2 * overlay < 1, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))

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
    k_specular: float = .5,
    k_shininess: float = 20,
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
    normal_vector = jnp.stack([jnp.real(normal_vector), jnp.imag(normal_vector), jnp.ones_like(normal_vector, dtype=jnp.float64)], axis=-1)
    normal_vector /= jnp.linalg.norm(normal_vector, axis=2, keepdims=True)

    diffuse_light = jnp.dot(normal_vector, light_vector)

    # reflection_vector is for the Phong reflection model, but we use the Blinn-Phong model instead
    # reflection_vector = 2 * jnp.dot(normal_vector, light_vector)[..., None] * normal_vector - light_vector[None, None, :]
    viewer_vector = jnp.array([0.0, 0.0, 1.0])
    halfway_vector = (light_vector + viewer_vector)
    halfway_vector /= jnp.linalg.norm(halfway_vector)
    specular_light = jnp.dot(normal_vector, halfway_vector)**k_shininess

    brightness = (
        k_ambient
        + k_diffuse * diffuse_light
        + k_specular * specular_light
    ) * opacity + (1 - opacity) / 2

    brightness = jnp.where(has_escaped, brightness, 0.0)

    return brightness


def create_light_source(azimuth: float = 90, elevation: float = 45) -> Array:
    """Creates a light source in 3D space.

    Args:
        azimuth (float): Azimuth of the light (0° = From bottom, 90° = From right)
        elevation (float): Elevation of the light (0° = Inside mandelbrot plane, 90° = From above)

    Returns:
        Array: Light source (normalized) vector
    """
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
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

def mandelbrot(width, height, max_iter, xlim, ylim, D=3):
    escape_radius = 10**5
    stripe_density = 2
    stripe_memory = 0.9

    xs = jnp.linspace(xlim[0], xlim[1], width * D, dtype=jnp.float64)
    ys = jnp.linspace(ylim[0], ylim[1], height * D, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, ys)

    c = X + 1j * Y
    z = jnp.zeros_like(c, dtype=jnp.complex128)
    dz_dc = jnp.ones_like(c, dtype=jnp.complex128)
    has_escaped = jnp.zeros(c.shape, dtype=bool)
    stripe_a = jnp.zeros(c.shape, dtype=jnp.float64)

    count = jnp.zeros(c.shape, dtype=jnp.int32)

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

    smooth_iter = count + smooth_iter
    smooth_iter *= has_escaped

    # Milnor distance normalization and contrast enhancement
    milnor_distance = -jnp.log(milnor_distance) / 12
    milnor_distance = 1/(1 + jnp.exp(-10 * (milnor_distance - 0.5)))

    brightness = brightness.reshape((height, D, width, D)).mean(axis=(1, 3))
    smooth_iter = smooth_iter.reshape((height, D, width, D)).mean(axis=(1, 3))
    milnor_distance = milnor_distance.reshape((height, D, width, D)).mean(axis=(1, 3))
    stripe_a = stripe_a.reshape((height, D, width, D)).mean(axis=(1, 3))

    return brightness, smooth_iter, milnor_distance, stripe_a

def apply_color(smooth_iter, stripe, milnor_distance, brightness, ncycle, rgb_theta=(.0, .15, .25)):
    # brightness = overlay(brightness, stripe) * (1 - milnor_distance) + milnor_distance * brightness

    transformed_iter = (jnp.sqrt(smooth_iter) % ncycle) / ncycle

    color = (1 + jnp.sin(2 * jnp.pi * (transformed_iter[..., None] + jnp.array(rgb_theta)[None, None, :]))) * 0.5
    color = overlay(color, brightness[..., None])
    return color

def gui_mandelbrot(xpixels):
    fig, ax = plt.subplots()

    ncycle = 32
    rgb_theta = (0.0, 0.15, 0.25)
    x_lim = (-2.6, 1.845)
    y_lim = (-1.25, 1.25)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]

    ypixels = round(xpixels / x_range * y_range)

    brightness, smooth_iter, milnor_distance, stripe = mandelbrot(xpixels, ypixels, 500, x_lim, y_lim, D=1)
    color = apply_color(smooth_iter, stripe, milnor_distance, brightness, np.sqrt(ncycle))
    img = ax.imshow(color)

    ax_r = plt.axes((0.05, 0.05, 0.15, 0.02))
    ax_g = plt.axes((0.05, 0.1, 0.15, 0.02))
    ax_b = plt.axes((0.05, 0.15, 0.15, 0.02))

    ax_oversample = plt.axes((0.4, 0.05, 0.2, 0.02))
    ax_maxiter = plt.axes((0.4, 0.1, 0.2, 0.02))
    ax_ncycle = plt.axes((0.4, 0.15, 0.2, 0.02))

    ax_x_center = plt.axes((0.75, 0.05, 0.2, 0.02))
    ax_y_center = plt.axes((0.75, 0.1, 0.2, 0.02))
    ax_zoom = plt.axes((0.75, 0.15, 0.2, 0.02))

    s_r = Slider(ax_r, '$\\theta_r$  ', 0, 1, valinit=rgb_theta[0], valstep=0.01, handle_style={'size': 7})
    s_g = Slider(ax_g, '$\\theta_g$  ', 0, 1, valinit=rgb_theta[1], valstep=0.01, handle_style={'size': 7})
    s_b = Slider(ax_b, '$\\theta_b$  ', 0, 1, valinit=rgb_theta[2], valstep=0.01, handle_style={'size': 7})

    s_oversample = Slider(ax_oversample, 'Sampling  ', 1, 4, valinit=1, valstep=1, handle_style={'size': 7})
    s_maxiter = Slider(ax_maxiter, 'Max Iter  ', 50, 2000, valinit=500, valstep=50, handle_style={'size': 7})
    s_ncycle = Slider(ax_ncycle, 'N-Cycle  ', 2, 64, valinit=ncycle, valstep=1, handle_style={'size': 7})

    s_x_center = Slider(ax_x_center, 'X  ', -3, 3, valinit=-0.5, valstep=0.001, handle_style={'size': 7})
    s_y_center = Slider(ax_y_center, 'Y  ', -2, 2, valinit=0, valstep=0.001, handle_style={'size': 7})
    s_zoom = Slider(ax_zoom, 'Zoom  ', 1, 50, valinit=1, valstep=0.1, handle_style={'size': 7})

    def update_mandelbrot(val, x_range, y_range):
        nonlocal brightness, smooth_iter, milnor_distance, stripe
        max_iter = int(s_maxiter.val)
        oversampling = int(s_oversample.val)
        x_center = s_x_center.val
        y_center = -s_y_center.val # Invert y-axis for intuitive zooming
        zoom = s_zoom.val

        new_x_range = x_range / zoom
        new_y_range = y_range / zoom
        new_x_lim = (x_center - new_x_range/2, x_center + new_x_range/2)
        new_y_lim = (y_center - new_y_range/2, y_center + new_y_range/2)

        brightness, smooth_iter, milnor_distance, stripe = mandelbrot(xpixels, ypixels, max_iter, new_x_lim, new_y_lim, D=oversampling)
        update_color(val)

    def update_color(val):
        rgb_theta = (s_r.val, s_g.val, s_b.val)
        ncycle = s_ncycle.val
        color = apply_color_jit(smooth_iter, stripe, milnor_distance, brightness, np.sqrt(ncycle), rgb_theta)
        img.set_data(color)
        fig.canvas.draw_idle()

    s_oversample.on_changed(lambda val: update_mandelbrot(val, x_range, y_range))
    s_maxiter.on_changed(lambda val: update_mandelbrot(val, x_range, y_range))
    s_x_center.on_changed(lambda val: update_mandelbrot(val, x_range, y_range))
    s_y_center.on_changed(lambda val: update_mandelbrot(val, x_range, y_range))
    s_zoom.on_changed(lambda val: update_mandelbrot(val, x_range, y_range))

    s_r.on_changed(update_color)
    s_g.on_changed(update_color)
    s_b.on_changed(update_color)
    s_ncycle.on_changed(update_color)

    ax.set_position((0, 0.3, 1, 0.7))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.show()

mandelbrot_step_jit = jax.jit(mandelbrot_step)
stripe_average_jit = jax.jit(stripe_average)
compute_lighting_jit = jax.jit(compute_lighting)
compute_smooth_iter_jit = jax.jit(compute_smooth_iter)
apply_color_jit = jax.jit(apply_color)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # Define the region of the complex plane to visualize
    x_lim = (-2.6, 1.845)
    y_lim = (-1.25, 1.25)
    # x_lim = (-0.5503295086752807, -0.5503293049351449)
    # y_lim = (-0.6259346555912755, -0.625934541001796)

    xpixels = 1080

    gui_mandelbrot(xpixels)
