# High-Performance 2D & 3D Convolution Engine

## Overview

This project implements a high-performance image and video processing engine focusing on convolution operations. It features a standalone C++ application for processing static images (2D) and video streams (3D/Temporal), as well as a Python wrapper that utilizes C++ shared libraries via `ctypes` for optimized performance.

The core functionality revolves around applying custom kernels (specifically Edge Detection) across spatial and temporal dimensions.

## Project Structure

* **`main.cpp`**: The primary C++ executable. Handles CLI arguments, image/video I/O using OpenCV, and implements multithreaded 3D convolution and FFT-based convolution.
* **`conv3d.cpp`**: A specialized C++ implementation of a 3-frame temporal convolution kernel, exported as `extern "C"` to be used as a shared library.
* **`conv.py`**: A Python script that acts as a wrapper. It captures video (file or webcam), manages threads, and passes image pointers to the compiled C++ library for processing.
* 
**`Makefile`**: Automation script for building the C++ executable, compiling shared libraries (`.so` / `.dll`), and running test suites.



## Dependencies

To build and run this project, you need the following installed:

**System Tools & Libraries:**

* **G++**: A modern C++ compiler supporting C++17 or later.
* **Make**: For build automation.
* **OpenCV (C++)**: Required for image and video I/O (`pkg-config --cflags --libs opencv4` is used in the Makefile).
* **FFTW3**: The code references a local build of FFTW 3.3.10 for Fast Fourier Transform operations.



**Python Libraries:**

* `numpy`
* `opencv-python`
* `scipy`

## Building the Project

The project includes a `Makefile` to handle compilation for different targets.

### 1. Build the Main C++ Application

To compile the standalone `main.cpp` executable with optimizations (`-O3 -march=native`):

```bash
make build
```



### 2. Build the Shared Library for Python

If you intend to use `conv.py`, you must compile the shared library first.

* **For Linux (`.so`):**
```bash
make c-ffi
```


* **For Windows (`.dll`):**
```bash
make c-dll
```


### 3. Clean Build Artifacts

To remove compiled binaries and output directories:

```bash
make clean
```

## Usage

### Using the C++ Application (`main`)

The compiled binary `build/main` supports processing single images or video files.

**Command Line Arguments:**

* `-i <path>`: Input file path.
* `-o <path>`: Output file path.
* `-verbose`: Enable detailed timing logs.

**Examples:**

```bash
# Process a single image (PPM or PNG)
./build/main -i ./input_images/lena.ppm -o ./output_images/lena_edge.ppm

# Process a video (MP4) using 3D Temporal Convolution
./build/main -i ./input_videos/sample.mp4 -o ./output_videos/processed.mp4
```

### Using the Python Wrapper (`conv.py`)

The Python script can run in two modes: **Real-time Camera** or **Video File**. This is determined by the `FFI` flag and platform checks inside the script.

1. Ensure you have built the shared library (`make c-ffi` or `make c-dll`).
2. Run the script:
```bash
python conv.py
```



**Configuration:**
You can modify the `__main__` block in `conv.py` to switch inputs:

```python
if __name__ == "__main__":
    kernel = K3D_EDGE_DET
    FFI = True
    # Windows defaults to Real-time Webcam
    if platform.system() == 'Windows':
        DLL_PATH = './build/conv3d.dll'
        main_real_time(kernel)
    # Linux defaults to Video File processing
    else:
        DLL_PATH = './build/libconv3d.so'
        main_video("./input_videos/sample.mp4", "./output_videos/output_py.mp4", kernel)
```

## Technical Details

### 3D Temporal Convolution

Unlike standard 2D convolution which operates on a single image plane (), this project implements **3D convolution**.

* **Depth:** The "Depth" dimension represents time. The kernel processes a stack of 3 frames simultaneously: **Previous**, **Current**, and **Next**.
* **Multithreading:** The C++ implementation detects hardware threads and divides the video processing workload into batches to maximize throughput.

### Foreign Function Interface (FFI)

The `conv.py` script bypasses the Global Interpreter Lock (GIL) for the heavy computational work. It passes pointers to raw Numpy array data directly to the C++ `conv3d_3p` function using `ctypes`. This allows Python to handle the high-level logic (I/O, threading) while C++ handles the pixel-level arithmetic.

### FFTW Integration

The `main.cpp` file contains an implementation of **FFT-based Convolution** using the FFTW3 library. This converts the image and kernel into the frequency domain to perform multiplication, which is mathematically equivalent to spatial convolution but can be significantly faster for large kernels.

---

*Note: Ensure the directory `./fftw-3.3.10/fftw-3.3.10/.libs/` exists and contains `libfftw3.a` as referenced in the Makefile, or adjust the `LIBS` path to match your system configuration.*