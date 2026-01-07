from __future__ import annotations
import time
import cv2
import ctypes
import numpy as np
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor
from typing import List
import platform
import numpy.typing as npt

FFI : bool = False
CONV_FUNC_NAME : str = 'conv3d_3p'
DLL_PATH : str = ''
NUM_THREADS : int = 12
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
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'FFI : {FFI}')
    print(f"Capture initialized: {w}x{h} at {fps} FPS, total frames: {total_frames}")

    return cap, fps, w, h, total_frames

def get_conv3d_func(dll_path: str, func_name: str):
    lib = ctypes.CDLL(dll_path)
    conv3d_func = lib[func_name]
    conv3d_func.argtypes = [
        ctypes.c_voidp,
        ctypes.c_voidp,
        ctypes.c_voidp,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_voidp,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    conv3d_func.restype = None
    return conv3d_func

def convert_frame(f):
    if FFI:
        return f.astype(np.float32)
    else:
        return f.astype(np.float32) / 255.0

def worker_conv(
    idx: int,
    conv3d_func: ctypes.CDLL,
    all_frames: List[npt.NDArray[np.float32]],
    kernel,
    width: int,
    height: int
) -> npt.NDArray[np.uint8]:
    frames = [
        all_frames[idx + 0],
        all_frames[idx + 1],
        all_frames[idx + 2],
    ]

    if FFI:
        out_frame = np.empty((height, width, 3), dtype=np.float32)

        p_prev = frames[0].ctypes.data_as(ctypes.c_voidp)
        p_curr = frames[1].ctypes.data_as(ctypes.c_voidp)
        p_next = frames[2].ctypes.data_as(ctypes.c_voidp)
        p_kern = k3d.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_out  = out_frame.ctypes.data_as(ctypes.c_voidp)

        conv3d_func(
            p_prev,
            p_curr,
            p_next,
            p_kern,
            p_out,
            width,
            height,
            kernel.shape[2],
            kernel.shape[1],
            kernel.shape[0],
            3
        )
        return np.clip(out_frame, 0, 255).astype(np.uint8)
    else:
        stacked = np.stack(frames, axis=0)  # shape (3, h, w, 3)
        out_channels = []
        for c in range(3):
            out = convolve(stacked[:, :, :, c], kernel, mode='nearest')
            out_channels.append(out[1])

        out_frame = np.stack(out_channels, axis=2)
        return np.clip(out_frame * 255, 0, 255).astype(np.uint8)

def main_real_time(k3d):

    conv3d_func = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    cap, fps, w, h, total_frames = init_capture(1)

    print("Press 'q' to quit.")
    frames = []
    for _ in range(2):
        ret, f = cap.read()
        if not ret: break
        frames.append(convert_frame(f))

    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(convert_frame(f))

        result_frame = worker_conv(
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

def main_video(input_path : str, output_path : str, k3d):

    conv3d_func = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    cap, fps, w, h, total_frames = init_capture(input_path)
    if total_frames < k3d.shape[0]:
        print("ERROR: Not enough frames in the video.")
        exit()
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    start_time = time.time()
    frames = []
    for _ in range(2):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(convert_frame(f))

    while True:
        future_frames: List[npt.NDArray[np.float32]] = []
        for _ in range(NUM_THREADS):
            ret, nf = cap.read()
            if not ret:
                break
            future_frames.append(convert_frame(nf))
        if len(future_frames) < NUM_THREADS:
            break

        all_frames = frames + future_frames

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [
                pool.submit(
                    worker_conv,
                    i,
                    conv3d_func,
                    all_frames,
                    k3d,
                    w,
                    h
                )
                for i in range(NUM_THREADS)
            ]
            results = [f.result() for f in futures]

        for i in range(NUM_THREADS):
            writer.write(results[i])
        for i in range(NUM_THREADS):
            frames.pop(0)
            frames.append(future_frames[i])

    print("All frames processed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f}s")

    cap.release()
    writer.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    k3d = K3D_EDGE_DET
    FFI = True
    if platform.system() == 'Windows':
        DLL_PATH = './build/conv3d.dll'
        main_real_time(k3d)
    else:
        DLL_PATH = './build/libconv3d.so'
        input_path = "./input_videos/sample.mp4"
        output_path = "./output_videos/output_py.mp4"
        main_video(input_path, output_path, k3d)
