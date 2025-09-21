import jax
from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton

from computation import mandelbrot, apply_color, apply_color_jit

class MandelbrotGUI:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # self.fig_slide, self.ax_slide = plt.subplots()
        self.ax.set_position((0, 0.25, 1, 0.75))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')

        # self.ax_slide.set_xticks([])
        # self.ax_slide.set_yticks([])
        # self.ax_slide.axis('off')

        self.x_lim = (-2.6, 1.845)
        self.y_lim = (-1.25, 1.25)

        self.x_range = self.x_lim[1] - self.x_lim[0]
        self.y_range = self.y_lim[1] - self.y_lim[0]

        self.xpixels = 1080
        self.ypixels = round(self.xpixels / self.x_range * self.y_range)

        self.default_xpixels = self.xpixels
        self.default_ypixels = self.ypixels
        self.default_x_lim = self.x_lim
        self.default_y_lim = self.y_lim

        self.max_iter = 500
        self.oversampling = 1
        self.ncycle = 32
        self.rgb_theta = (0.0, 0.15, 0.25)
        self.stripe_enabled = False

        self.brightness, self.smooth_iter, self.milnor_distance, self.stripe = mandelbrot(
            self.xpixels, self.ypixels, self.max_iter, self.x_lim, self.y_lim, D=self.oversampling
        )

        self.color = apply_color_jit(
            self.smooth_iter, self.stripe, self.stripe_enabled, self.milnor_distance, self.brightness, np.sqrt(self.ncycle), self.rgb_theta
        )

        self.img: AxesImage = self.ax.imshow(self.color)

        # Add sliders and controllers
        self.add_slider()
        self.add_button()
        self.selector = RectangleSelector(self.ax, self.on_rectangle_select, useblit=True, button=[MouseButton.LEFT],
                                   minspanx=5, minspany=5, spancoords='pixels')
        plt.show()


    def add_slider(self):
        ax_r = plt.axes((0.15, 0.1, 0.15, 0.02))
        ax_g = plt.axes((0.15, 0.15, 0.15, 0.02))
        ax_b = plt.axes((0.15, 0.2, 0.15, 0.02))
        ax_ncycle = plt.axes((0.15, 0.05, 0.15, 0.02))

        ax_oversample = plt.axes((0.5, 0.2, 0.2, 0.02))
        ax_maxiter = plt.axes((0.5, 0.15, 0.2, 0.02))
        ax_status = plt.axes((0.5, 0.05, 0.1, 0.02))

        self.status_text = ax_status.text(0.5, 0.5, 'Done.', ha='center', va='center', transform=ax_status.transAxes)
        ax_status.set_xticks([])
        ax_status.set_yticks([])
        ax_status.axis('off')

        self.s_r = Slider(ax_r, '$\\theta_r$  ', 0, 1, valinit=self.rgb_theta[0], valstep=0.01, handle_style={'size': 7})
        self.s_g = Slider(ax_g, '$\\theta_g$  ', 0, 1, valinit=self.rgb_theta[1], valstep=0.01, handle_style={'size': 7})
        self.s_b = Slider(ax_b, '$\\theta_b$  ', 0, 1, valinit=self.rgb_theta[2], valstep=0.01, handle_style={'size': 7})
        self.s_ncycle = Slider(ax_ncycle, 'N-Cycle  ', 2, 64, valinit=self.ncycle, valstep=1, handle_style={'size': 7})

        self.s_oversample = Slider(ax_oversample, 'Sampling  ', 1, 4, valinit=self.oversampling, valstep=1, handle_style={'size': 7})
        self.s_maxiter = Slider(ax_maxiter, 'Max Iter  ', 50, 2000, valinit=self.max_iter, valstep=50, handle_style={'size': 7})

        self.s_r.on_changed(self.update_color)
        self.s_g.on_changed(self.update_color)
        self.s_b.on_changed(self.update_color)
        self.s_ncycle.on_changed(self.update_color)

        self.s_oversample.on_changed(self.update_parameters)
        self.s_maxiter.on_changed(self.update_parameters)

    def add_button(self):
        def reset_view(val):
            self.xpixels = self.default_xpixels
            self.ypixels = self.default_ypixels
            self.x_lim = self.default_x_lim
            self.y_lim = self.default_y_lim
            self.x_range = self.x_lim[1] - self.x_lim[0]
            self.y_range = self.y_lim[1] - self.y_lim[0]
            self.compute_mandelbrot()

        def enable_selection(event):
            self.selector.active = not self.selector.active
            self.fig.canvas.draw_idle()

        def swap_color(button: Button, state):
            if state:
                button.color = "#0aa815"
                button.hovercolor = "#07780f"
            else:
                button.color = "#d80d0d"
                button.hovercolor = "#a80a0a"
            return

        def toggle_stripe(event):
            self.stripe_enabled = not self.stripe_enabled
            self.update_color(None)

        ax_reset = plt.axes((0.8, 0.02, 0.1, 0.04))
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(reset_view)

        ax_enable_selection = plt.axes((0.8, 0.09, 0.1, 0.04))
        self.enable_selection_button = Button(ax_enable_selection, 'Select', color="#0aa815", hovercolor="#07780f")
        self.enable_selection_button.on_clicked(enable_selection)
        self.enable_selection_button.on_clicked(lambda event: swap_color(self.enable_selection_button, self.selector.active))

        ax_stripe = plt.axes((0.8, 0.15, 0.1, 0.04))
        self.stripe_button = Button(ax_stripe, 'Stripe', color="#d80d0d", hovercolor="#a80a0a")
        self.stripe_button.on_clicked(toggle_stripe)
        self.stripe_button.on_clicked(lambda event: swap_color(self.stripe_button, self.stripe_enabled))

    def update_color(self, val):
        self.rgb_theta = (self.s_r.val, self.s_g.val, self.s_b.val)
        self.ncycle = self.s_ncycle.val

        color = apply_color_jit(self.smooth_iter, self.stripe, self.stripe_enabled, self.milnor_distance, self.brightness, np.sqrt(self.ncycle), self.rgb_theta)

        self.img.set_data(color)
        self.fig.canvas.draw_idle()

    def update_parameters(self, val):
        self.oversampling = int(self.s_oversample.val)
        self.max_iter = int(self.s_maxiter.val)
        self.compute_mandelbrot()

    def on_rectangle_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        if x1 == x2 or y1 == y2:
            return

        old_ratio = self.x_range / self.y_range

        self.x_lim = (x1 / self.xpixels * self.x_range + self.x_lim[0], x2 / self.xpixels * self.x_range + self.x_lim[0])
        self.y_lim = (y1 / self.ypixels * self.y_range + self.y_lim[0], y2 / self.ypixels * self.y_range + self.y_lim[0])
        self.x_lim = (min(self.x_lim), max(self.x_lim))
        self.y_lim = (min(self.y_lim), max(self.y_lim))

        self.x_range = self.x_lim[1] - self.x_lim[0]
        self.y_range = self.y_lim[1] - self.y_lim[0]

        new_ratio = self.x_range / self.y_range

        self.xpixels = int(self.xpixels * new_ratio / old_ratio)
        self.ypixels = int(self.ypixels * old_ratio / new_ratio)

        self.compute_mandelbrot()

    def compute_mandelbrot(self):
        self.set_status('Computing...', 'red')

        self.brightness, self.smooth_iter, self.milnor_distance, self.stripe = mandelbrot(
            self.xpixels, self.ypixels, int(self.s_maxiter.val), self.x_lim, self.y_lim, D=int(self.s_oversample.val)
        )

        self.set_status('Done.', 'green')

        self.update_color(None)

    def set_status(self, text: str, color):
        self.status_text.set_text(text)
        self.status_text.set_color(color)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print(f"JAX accelerator in use: {jax.devices()[0].device_kind}")
    MandelbrotGUI()
