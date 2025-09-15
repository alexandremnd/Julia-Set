# Julia Set & Mandelbrot Set Drawer

This project uses Python and JAX (or numpy) to render high-definition Julia and Mandelbrot sets with optional shading and coloring.

## Installation

### 1. Initialize the Environment
Using `uv`, in the project directory, init a virtual environment with:
```sh
uv venv
```
or with any virtual environment manager.

### 2. Install Dependencies
To install all required packages, run:
```sh
uv pip install .
```
Or, if you don't use `uv`:
```sh
pip install -r requirements.txt
```

### 4. Activate the Environment
To activate the environment (assuming it is in `./.venv/`):
```sh
source .venv/bin/activate
```

### 5. Run the Project
You can run the main script with:
```sh
python main.py
```