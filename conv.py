from __future__ import annotations
import math
import shutil
import os
import sys
import time
import cv2
import ctypes
import argparse
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor
from typing import List
import enum
try: import torch
except ImportError: pass

class ConvMode(enum.Enum):
    PY_NESTED_LOOPS = 1
    PY_NESTED_LOOPS_VECTORIZED = 2
    PY_SCIPY_CONV = 3
    PY_SCIPY_CONV_MT = 4
    PY_OCV_FILT2D = 5
    PY_OCV_FILT2D_MT = 6
    PY_TORCH_CONV3D = 7

CONV_FUNC_NAME : str = 'conv3d'
DLL_PATH : str = './build/libconv3d.so'
K3D_EDGE_DET = np.array([
    [
        [-1, -2, -1],
        [-2, -4, -2],
        [-1, -2, -1],
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ],
], dtype=np.float32)

def init_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Capture initialized: {w}x{h} at {fps} FPS, total frames: {total_frames}")

    return cap, fps, w, h, total_frames

def get_conv3d_func(dll_path: str, func_name: str):
    lib = ctypes.CDLL(dll_path)
    conv3d_func = lib[func_name]
    conv3d_func.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    conv3d_func.restype = None
    return conv3d_func


def conv3d_pytorch(frames, k3d, cs):
    fd = len(frames)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stacked_frames = np.stack(frames, axis=0)
    
    # Shape: (1_batch, Channels, Depth, Height, Width)
    img_tensor = torch.from_numpy(stacked_frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    
    # Flip kernel for true convolution matching
    k3d_flipped = np.flip(k3d, axis=(0, 1, 2)).copy()
    k_tensor = torch.from_numpy(k3d_flipped).unsqueeze(0).unsqueeze(0).to(device)
    k_tensor = k_tensor.repeat(cs, 1, 1, 1, 1)

    with torch.no_grad():
        out_tensor = torch.nn.functional.conv3d(
            img_tensor, 
            k_tensor, 
            padding='same', 
            groups=cs
        )
    
    # Remove batch dim and permute back to (Depth, Height, Width, Channels)
    out_numpy = out_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    
    return out_numpy

def conv3d__(
    frames: List[npt.NDArray[np.float32]],
    k3d: npt.NDArray[np.float32],
    tc: int,
    mode: ConvMode
) -> List[npt.NDArray[np.float32]]:
    
    kd, kh, kw = k3d.shape
    w = frames[0].shape[1]
    h = frames[0].shape[0]
    d = len(frames)
    cs = frames[0].shape[2]
    
    out_frames = np.zeros((d, h, w, cs), dtype=np.float32)
    k3d = np.flip(k3d, axis=(0, 1, 2))
    
    padD = kd // 2
    padH = kh // 2
    padW = kw // 2

    if mode == ConvMode.PY_NESTED_LOOPS:
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    for c in range(cs):
                        accum = 0.0
                        for dz in range(kd):
                            for dy in range(kh):
                                for dx in range(kw):
                                    z_in = z + dz - padD
                                    y_in = y + dy - padH
                                    x_in = x + dx - padW
                                    
                                    if 0 <= z_in < d and 0 <= y_in < h and 0 <= x_in < w:
                                        pixel_val = frames[z_in][y_in, x_in, c]
                                        k_dz = kd - 1 - dz
                                        k_dy = kh - 1 - dy
                                        k_dx = kw - 1 - dx
                                        accum += pixel_val * k3d[k_dz, k_dy, k_dx]
                        out_frames[z, y, x, c] = accum

    elif mode == ConvMode.PY_NESTED_LOOPS_VECTORIZED:
        stacked = np.stack(frames, axis=0)
        padded = np.pad(stacked, ((padD, padD), (padH, padH), (padW, padW), (0, 0)), mode='constant', constant_values=0.0)
        
        for dz in range(kd):
            for dy in range(kh):
                for dx in range(kw):
                    k_dz = kd - 1 - dz
                    k_dy = kh - 1 - dy
                    k_dx = kw - 1 - dx
                    weight = k3d[k_dz, k_dy, k_dx]
                    
                    if weight == 0.0: continue
                    
                    shifted_view = padded[dz : dz + d, dy : dy + h, dx : dx + w, :]
                    out_frames += shifted_view * weight

    elif mode == ConvMode.PY_SCIPY_CONV:
        stacked = np.stack(frames, axis=0)
        for c in range(cs):
            out_frames[:, :, :, c] = convolve(
                stacked[:, :, :, c],
                k3d,
                mode='constant',
                cval=0.0
            )

    elif mode == ConvMode.PY_SCIPY_CONV_MT:
        stacked = np.stack(frames, axis=0)
        with ThreadPoolExecutor(max_workers=tc) as pool:
            futures = [
                pool.submit(convolve, stacked[:, :, :, c], k3d, mode='constant', cval=0.0)
                for c in range(cs)
            ]
            for c, f in enumerate(futures):
                out_frames[:, :, :, c] = f.result()

    elif mode == ConvMode.PY_OCV_FILT2D:
        for z in range(d):
            for dz in range(kd):
                z_in = z + dz - padD
                if 0 <= z_in < d:
                    k_flipped = cv2.flip(k3d[kd - 1 - dz], -1)
                    filtered = cv2.filter2D(
                        frames[z_in], 
                        cv2.CV_32F, 
                        k_flipped, 
                        borderType=cv2.BORDER_CONSTANT
                    )
                    out_frames[z] += filtered.reshape(h, w, cs)

    elif mode == ConvMode.PY_OCV_FILT2D_MT:
        def process_single_out_frame(z):
            acc = np.zeros((h, w, cs), dtype=np.float32)
            for dz in range(kd):
                z_in = z + dz - padD
                if 0 <= z_in < d:
                    k_flipped = cv2.flip(k3d[kd - 1 - dz], -1)
                    filtered = cv2.filter2D(
                        frames[z_in], 
                        cv2.CV_32F, 
                        k_flipped, 
                        borderType=cv2.BORDER_CONSTANT
                    )
                    acc += filtered.reshape(h, w, cs)
            return z, acc

        with ThreadPoolExecutor(max_workers=tc) as pool:
            futures = [pool.submit(process_single_out_frame, z) for z in range(d)]
            for f in futures:
                z_idx, acc_frame = f.result()
                out_frames[z_idx] = acc_frame

    elif mode == ConvMode.PY_TORCH_CONV3D:
        if 'torch' not in sys.modules: raise ImportError("PyTorch not installed.")
        out_frames = conv3d_pytorch(frames, k3d, cs)

    else:
        raise ValueError("Invalid convolution mode specified.")

    return [out_frames[i] for i in range(d)]

def conv3d(
    idx: int,
    conv3d_func: ctypes.CDLL,
    all_frames: List[npt.NDArray[np.float32]],
    k3d : npt.NDArray[np.float32],
    tc,
    mode: ConvMode) -> npt.NDArray[np.uint8]:

    w = all_frames[0].shape[1]
    h = all_frames[0].shape[0]
    d = len(all_frames)
    cs = all_frames[0].shape[2]
    kd, kh, kw = k3d.shape

    frames = all_frames[idx:idx+k3d.shape[0]]
    if len(frames) < k3d.shape[0]:
        frames += [np.zeros((h, w, cs), dtype=np.float32)] * (k3d.shape[0] - len(frames))

    out_frame = conv3d__(frames, k3d, tc, mode)

    # padH = kh // 2
    # padW = kw // 2
    # if padH > 0:
    #     out_frame[:padH, :] = 0.0
    #     out_frame[-padH:, :] = 0.0
    # if padW > 0:
    #     out_frame[:, :padW] = 0.0
    #     out_frame[:, -padW:] = 0.0
    return np.clip(out_frame, 0, 255).astype(np.uint8)

    # c_float_p = ctypes.POINTER(ctypes.c_float)
    # FloatPtrArray3 = c_float_p * k3d.shape[0]
    # inputs = FloatPtrArray3()
    # for i in range(k3d.shape[0]): inputs[i] = safe_frames[i].ctypes.data_as(c_float_p)
    # p_kern = safe_k3d.ctypes.data_as(c_float_p)
    # out_frame = np.zeros((height, width, cs), dtype=np.float32)
    # p_out  = out_frame.ctypes.data_as(c_float_p)

    # conv3d_func(
    #     inputs,
    #     p_kern,
    #     p_out,

    #     width,
    #     height,
    #     len(safe_frames),

    #     k3d.shape[2],
    #     k3d.shape[1],
    #     k3d.shape[0],

    #     cs,
    #     tc,
    # )
    # return np.clip(out_frame, 0, 255).astype(np.uint8)

def conv3d_on_video(input_path : str, output_path : str, k3d : npt.NDArray[np.float32], enable_writer : bool, tc : int, mode: ConvMode):

    conv3d_func = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    cap, fps, w, h, total_frames = init_capture(input_path)
    if total_frames < k3d.shape[0]:
        print("ERROR: Not enough frames in the video.")
        sys.exit()
    
    name, ext = os.path.splitext(output_path)
    output_path = f"{name}_{k3d.shape[0]}x{k3d.shape[1]}x{k3d.shape[2]}_tc_{tc}_{mode.value}_{mode.name}{ext}"
    writer = None
    if enable_writer:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frames = []
    for _ in range(k3d.shape[0] - 1):
        ret, f = cap.read()
        if not ret: break
        frames.append(f.astype(np.float32))

    start_time = time.time()
    while True:
        for _ in range(tc):
            ret, f = cap.read()
            if not ret: break
            frames.append(f.astype(np.float32))
        if len(frames) < tc: break

        with ThreadPoolExecutor(max_workers=tc) as pool:
            futures = [
                pool.submit(
                    conv3d,
                    i,
                    conv3d_func,
                    frames,
                    k3d,
                    tc,
                    mode
                )
                for i in range(tc)
            ]
            results = [f.result() for f in futures]
        if enable_writer:
            for i in range(tc):
                if results[i] is not None: writer.write(results[i])
        frames = frames[tc:]
    end_time = time.time()

    cap.release()
    if enable_writer:
        writer.release()
        print(f"Saved to {output_path}")

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f}s")
    return elapsed_time


def generate_3d_prewitt_z(size: int, normalize: bool = True) -> np.ndarray:
    if size % 2 == 0 or size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3.")
    K = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2
    for z in range(size):
        K[z, :, :] = (z - center)
    if normalize:
        positive_sum = K[K > 0].sum()
        if positive_sum > 0:
            K /= positive_sum
    return K

def test_functionality():
    w = 5
    h = 5
    d = 5
    cs = 1
    tc = 1
    dim = 3
    test_frames = []
    for i in range(d):
        frame = np.zeros((h, w, cs), dtype=np.float32)
        for c in range(cs):
            for y in range(h):
                for x in range(w):
                    frame[y, x, c] = i * h * w * cs + c * h * w + y * w + x + 1
        test_frames.append(frame)
    test_kernel = np.zeros((dim, dim, dim), dtype=np.float32)
    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
                test_kernel[z, y, x] = z * dim * dim + y * dim + x + 1
    modes = list(ConvMode)
    results = {}
    eps = 1e-3
    times_taken = {}
    for m in modes:
        print(f"Testing mode: {m.name}")
        start_time = time.time()
        result = conv3d__(test_frames, test_kernel, tc, m)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times_taken[m] = elapsed_time
        results[m] = result
        print('-' * 30)
    for i in range(1, len(modes)):
        if not np.allclose(results[modes[0]], results[modes[i]], atol=eps):
            return False

    print('the result:')
    res = results[modes[0]]
    w = res[0].shape[1]
    h = res[0].shape[0]
    d = len(res)
    for z in range(d):
        for y in range(h):
            for x in range(w):
                print(f"{res[z][y, x, 0]:<10.2f}", end=' ')
            print()
        print('-' * 30)
    return True

def test_video(input_path, output_path, k3d):
    all_modes = list(ConvMode)
    tc = 4
    with ThreadPoolExecutor(max_workers=len(all_modes)) as pool:
        futures = [
            pool.submit(
                conv3d_on_video,
                input_path,
                output_path,
                k3d,
                True,
                tc,
                mode=mode
            )
            for mode in all_modes
        ]
        for future in futures: future.result()
    return

def test_perf():
    dims = [3, 5, 7, 9]
    tc_values = list(range(1, (2 * os.cpu_count()) + 1))
    modes = list(ConvMode)
    results = {mode: {dim: [] for dim in dims} for mode in modes}
    cs = 1
    w = 100
    h = 100
    d = 100
    test_frames = [np.random.rand(h, w, cs).astype(np.float32) * 255 for _ in range(d)]
    for dim in dims:
        test_kernel = generate_3d_prewitt_z(dim, normalize=False)
        for tc in tc_values:
            for mode in modes:
                print(f"dim: {dim}, thread count: {tc}, Testing mode: {mode.name}")
                if tc > 1 and mode in [ConvMode.PY_NESTED_LOOPS]:
                    print(f"Skipping {mode.name} for thread count {tc} since it's not designed for multi-threading.")
                    results[mode][dim].append(results[mode][dim][0])
                    continue
                start_time = time.time()
                conv3d__(test_frames, test_kernel, tc, mode)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # elapsed_time = conv3d_on_video("./input_videos/sample.mp4", "./output_videos/perf_test.mp4", k3d, False, tc, mode)
                results[mode][dim].append(elapsed_time)

    if os.path.exists('./metrics'): shutil.rmtree('./metrics')
    os.makedirs('./metrics/mode-based', exist_ok=True)
    os.makedirs('./metrics/dimension-based', exist_ok=True)
    os.makedirs('./metrics/thread-count-based', exist_ok=True)
    for mode in modes:
        plt.figure(figsize=(10, 6))
        for dim in dims:
            plt.plot(tc_values, results[mode][dim], label=f'Dim {dim}x{dim}x{dim}', marker='o')
        plt.title(f'Performance of {mode.name}')
        plt.xlabel('Thread Count')
        plt.ylabel('Time (seconds)')
        plt.xticks(tc_values)
        plt.grid()
        plt.legend()
        plt.savefig(f'./metrics/mode-based/{mode.name}.png')
        plt.close()

    alone_list = [ConvMode.PY_NESTED_LOOPS, ConvMode.PY_SCIPY_CONV]
    for dim in dims:
        plt.figure(figsize=(10, 6))
        for mode in modes:
            if mode in alone_list: continue
            plt.plot(tc_values, results[mode][dim], label=f'{mode.name}', marker='o')
        plt.title(f'Performance Comparison for Kernel Dim {dim}x{dim}x{dim}')
        plt.xlabel('Thread Count')
        plt.ylabel('Time (seconds)')
        plt.xticks(tc_values)
        plt.grid()
        plt.legend()
        plt.savefig(f'./metrics/dimension-based/{dim}.png')
        plt.close()

        for mode in alone_list:
            plt.figure(figsize=(10, 6))
            plt.plot(tc_values, results[mode][dim], label=f'{mode.name}', marker='o')
            plt.title(f'Performance of {mode.name} for Kernel Dim {dim}x{dim}x{dim}')
            plt.xlabel('Thread Count')
            plt.ylabel('Time (seconds)')
            plt.xticks(tc_values)
            plt.grid()
            plt.legend()
            os.makedirs(f'./metrics/dimension-based/alone/{mode.name}', exist_ok=True)
            plt.savefig(f'./metrics/dimension-based/alone/{mode.name}/{dim}.png')
            plt.close()

    for tc in tc_values:
        plt.figure(figsize=(10, 6))
        for mode in modes:
            if mode in alone_list: continue
            times_for_tc = [results[mode][dim][tc - 1] for dim in dims]
            plt.plot(dims, times_for_tc, label=f'{mode.name}', marker='o')
        plt.title(f'Performance Comparison for Thread Count {tc}')
        plt.xlabel('Kernel Dimension')
        plt.ylabel('Time (seconds)')
        plt.xticks(dims)
        plt.grid()
        plt.legend()
        plt.savefig(f'./metrics/thread-count-based/{tc}.png')
        plt.close()

        for mode in alone_list:
            plt.figure(figsize=(10, 6))
            times_for_tc = [results[mode][dim][tc - 1] for dim in dims]
            plt.plot(dims, times_for_tc, label=f'{mode.name}', marker='o')
            plt.title(f'Performance of {mode.name} for Thread Count {tc}')
            plt.xlabel('Kernel Dimension')
            plt.ylabel('Time (seconds)')
            plt.xticks(dims)
            plt.grid()
            plt.legend()
            os.makedirs(f'./metrics/thread-count-based/alone/{mode.name}', exist_ok=True)
            plt.savefig(f'./metrics/thread-count-based/alone/{mode.name}/{tc}.png')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run 3D Convolution on Video")
    parser.add_argument('--input-path', type=str, dest='input_path', help="Path to input video")
    parser.add_argument('--output-path', type=str, dest='output_path', help="Path to output video")
    parser.add_argument('--test-modes', action='store_true', dest='test_modes', help="Run functionality tests for all convolution modes and exit")
    parser.add_argument('--test-video', action='store_true', dest='test_video', help="Run tests on sample videos and exit")
    parser.add_argument('--test-perf', action='store_true', dest='perf_test', help="Run performance tests for conv3d and exit")

    args = parser.parse_args()
    
    if args.test_modes:
        if test_functionality(): print("All convolution modes produce approximately the same output. Functionality test passed.")
        else: print("Convolution modes produce different outputs. Functionality test failed.")
        sys.exit(0)
    elif args.test_video:
        k3d = generate_3d_prewitt_z(3, normalize=False)
        test_video(args.input_path, args.output_path, k3d)
        sys.exit(0)
    elif args.perf_test:
        test_perf()
        sys.exit(0)

if __name__ == "__main__":
    main()


"""

def test_conv3d__():
    global FFI
    enable_writer = False
    dims = [3, 5, 7, 9]
    norm = False
    input_path = "./input_videos/sample.mp4"
    output_path = "./output_videos/output_py.mp4"
    for dim in dims:
        kernel = generate_3d_prewitt_z(dim, norm)
        for tc in range(1, (2 * os.cpu_count()) + 1):
            print('-' * 50)
            FFI = True
            conv3d_on_video(input_path, output_path, kernel, enable_writer)
            print('-' * 15)
            FFI = False
            conv3d_on_video(input_path, output_path, kernel, enable_writer)
            print('-' * 50)

def test_conv3d():
    global FFI
    runs = 5
    ffi_conv3d = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    dims = [3, 5, 7, 9, 11]
    results_ffi = [[0 for _ in range(2 * os.cpu_count())] for _ in range(len(dims))]
    results_non_ffi = [[0 for _ in range(2 * os.cpu_count())] for _ in range(len(dims))]
    norm = False
    w, h = 1280, 720
    print(f"Testing with random frames of size {w}x{h} and kernels of dimensions: {dims}")
    frames = [np.random.rand(h, w, 3).astype(np.float32) * 255 for _ in range(11)]
    for _ in range(runs):
        for dim in dims:
            kernel = generate_3d_prewitt_z(dim, norm)
            for tc in range(1, (2 * os.cpu_count()) + 1):
                FFI = True
                start_time = time.time()
                conv3d(0, ffi_conv3d, frames, kernel, w, h, tc)
                end_time = time.time()
                elapsed_time = end_time - start_time
                results_ffi[dims.index(dim)][tc - 1] += elapsed_time

                FFI = False
                start_time = time.time()
                conv3d(0, ffi_conv3d, frames, kernel, w, h, tc)
                end_time = time.time()
                elapsed_time = end_time - start_time
                results_non_ffi[dims.index(dim)][tc - 1] += elapsed_time
    
    for i in range(len(dims)): results_ffi[i] = [t / runs for t in results_ffi[i]]
    for i in range(len(dims)): results_non_ffi[i] = [t / runs for t in results_non_ffi[i]]

    cols = 2
    rows = math.ceil(len(dims) / cols)
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(dims):
        plt.subplot(rows, cols, i + 1)
        tc_values = list(range(1, (2 * os.cpu_count()) + 1))
        ffi_times = results_ffi[i]
        plt.plot(tc_values, ffi_times, label='FFI', marker='o')
        non_ffi_times = results_non_ffi[i]
        plt.plot(tc_values, non_ffi_times, label='Non-FFI', marker='o')
        plt.title(f'Kernel Size: {dim}x{dim}x{dim}')
        plt.xlabel('Thread Count')
        plt.ylabel('Time (seconds)')
        plt.xticks(tc_values)
        plt.grid()
        plt.legend()
    plt.suptitle('3D Convolution Time vs Thread Count')
    plt.tight_layout()
    plt.savefig('./conv3d_performance.png')
    # plt.show()

def main_real_time(k3d):

    conv3d_func = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    cap, fps, w, h, total_frames = init_capture(1)

    print("Press 'q' to quit.")
    frames = []
    for _ in range(2):
        ret, f = cap.read()
        if not ret: break
        frames.append(f.astype(np.float32))

    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f.astype(np.float32))

        result_frame = conv3d(
            0,
            conv3d_func,
            frames,
            k3d,
            w,
            h
        )
        result_frame = cv2.flip(result_frame, 1)
        cv2.imshow("3D Conv in Real-Time", result_frame)
        frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""