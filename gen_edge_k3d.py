import numpy as np

def generate_3d_z_sobel(size: int) -> np.ndarray:
    """
    Generates an extended 3D Sobel kernel for Z-axis edge detection.
    """
    if size % 2 == 0 or size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3.")
        
    # 1. 1D Spatial Smoothing (Pascal's triangle)
    S = np.array([1.0], dtype=np.float32)
    for _ in range(size - 1):
        S = np.convolve(S, [1.0, 1.0])
        
    # 2. 1D Temporal/Depth Derivative
    # Calculated by smoothing the standard [-1, 1] difference
    smooth_for_deriv = np.array([1.0], dtype=np.float32)
    for _ in range(size - 2):
        smooth_for_deriv = np.convolve(smooth_for_deriv, [1.0, 1.0])
    D = np.convolve(smooth_for_deriv, [-1.0, 1.0])
    
    # 3. 2D Spatial Matrix (Outer product of X and Y smoothing)
    M = np.outer(S, S)
    
    # 4. Construct the full 3D Kernel
    K = np.zeros((size, size, size), dtype=np.float32)
    for z in range(size):
        K[z, :, :] = D[z] * M
        
    return K
