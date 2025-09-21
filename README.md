# Mandelbrot Set Explorer

<!-- ![Mandelbrot Example](https://upload.wikimedia.org/wikipedia/commons/2/21/Mandel_zoom_00_mandelbrot_set.jpg) -->

A high-performance, interactive Mandelbrot and Julia set explorer written in Python, leveraging JAX for GPU/TPU acceleration and Matplotlib for a modern, responsive GUI. Features advanced coloring, lighting, and real-time parameter adjustment.


## Features
- **Ultra-fast rendering** with JAX (CPU/GPU/TPU support - no Apple GPU support sadly ...)
- **Interactive GUI**: zoom, adjust color and fractal parameters live
- **Advanced coloring**: smooth iteration, stripe average and Blinn-Phong lighting
- **Custom color palettes**: continuous color table modes
- **High-definition output**: anti-aliasing via oversampling

<!-- Centered GIF for GitHub markdown (HTML style is ignored by GitHub, so use table hack) -->
<p align="center">
  <img src="img/usage.gif" width="500"/>
</p>



## Quickstart

1. Install dependencies
```sh
uv pip install .
```

2. Run the interactive explorer
```sh
uv run main.py
```

3. Controls
- **Sliders**: Adjust color phase, oversampling, max iterations, color cycles, zoom, and center
- **Live update**: All changes are reflected instantly (or not if you are on CPU)


## Credits
- Original repo from [jlesuffleur](https://github.com/jlesuffleur/gpu_mandelbrot): used milnor distance/stripe averaging and color palette from his repo. This project was mainly an opportunity to experiment with JAX library.


## License
This project is licensed under the MIT license - see [LICENSE](LICENSE) file for details.
