from __future__ import annotations
import os
import time
import cv2
import ctypes
import numpy as np
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor
from typing import List
import platform
import numpy.typing as npt

# FFI : bool = True
FFI : bool = False
CONV_FUNC_NAME : str = 'conv3d'
DLL_PATH : str = './build/libconv3d.so'
THREAD_COUNT : int = 12
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
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
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


def worker_conv3d(
    idx: int,
    conv3d_func: ctypes.CDLL,
    all_frames: List[npt.NDArray[np.float32]],
    k3d : npt.NDArray[np.float32],
    width: int,
    height: int,
) -> npt.NDArray[np.uint8]:
    cs : int = 3
    if FFI:
        c_float_p = ctypes.POINTER(ctypes.c_float)
        FloatPtrArray3 = c_float_p * k3d.shape[0]

        frames = all_frames[idx:idx+k3d.shape[0]]
        safe_frames = [np.ascontiguousarray(f, dtype=np.float32) for f in frames]
        safe_k3d = np.ascontiguousarray(k3d, dtype=np.float32)

        inputs = FloatPtrArray3()
        for i in range(k3d.shape[0]): inputs[i] = safe_frames[i].ctypes.data_as(c_float_p)
        p_kern = safe_k3d.ctypes.data_as(c_float_p)
        out_frame = np.zeros((height, width, cs), dtype=np.float32)
        p_out  = out_frame.ctypes.data_as(c_float_p)

        conv3d_func(
            inputs,
            p_kern,
            p_out,

            width,
            height,
            len(safe_frames),

            k3d.shape[2],
            k3d.shape[1],
            k3d.shape[0],

            cs
        )
        return np.clip(out_frame, 0, 255).astype(np.uint8)
    else:
        stacked = np.stack(all_frames[idx:idx+k3d.shape[0]], axis=0)  # shape = (k3d.shape[0], height, width, cs)
        out_channels = []
        kh, kw = k3d.shape[1], k3d.shape[2]
        padH = kh // 2
        padW = kw // 2
        
        for c in range(cs):
            out = convolve(stacked[:, :, :, c], k3d, mode='constant', cval=0.0)
            channel_out = out[1].copy()
            if padH > 0:
                channel_out[:padH, :] = 0.0
                channel_out[-padH:, :] = 0.0
            if padW > 0:
                channel_out[:, :padW] = 0.0
                channel_out[:, -padW:] = 0.0
            out_channels.append(channel_out)
        out_frame = np.stack(out_channels, axis=2)
        return np.clip(out_frame, 0, 255).astype(np.uint8)

def conv3d_on_video(input_path : str, output_path : str, k3d : npt.NDArray[np.float32]):

    conv3d_func = get_conv3d_func(DLL_PATH, CONV_FUNC_NAME)
    cap, fps, w, h, total_frames = init_capture(input_path)
    if total_frames < k3d.shape[0]:
        print("ERROR: Not enough frames in the video.")
        exit()
    
    name, ext = os.path.splitext(output_path)
    output_path = f"{name}_FFI_{FFI}{ext}"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frames = []
    for _ in range(k3d.shape[0] - 1):
        ret, f = cap.read()
        if not ret: break
        frames.append(f.astype(np.float32))

    start_time = time.time()
    while True:
        for _ in range(THREAD_COUNT):
            ret, f = cap.read()
            if not ret: break
            frames.append(f.astype(np.float32))
        if len(frames) < THREAD_COUNT: break

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as pool:
            futures = [
                pool.submit(
                    worker_conv3d,
                    i,
                    conv3d_func,
                    frames,
                    k3d,
                    w,
                    h,
                )
                for i in range(THREAD_COUNT)
            ]
            results = [f.result() for f in futures]

        for i in range(THREAD_COUNT): writer.write(results[i])
        frames = frames[THREAD_COUNT:]

    end_time = time.time()
    print("All frames processed.")
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f}s")

    cap.release()
    writer.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    kernel = K3D_EDGE_DET
    input_path = "./input_videos/sample.mp4"
    output_path = "./output_videos/output_py.mp4"
    conv3d_on_video(input_path, output_path, kernel)


"""
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

        result_frame = worker_conv3d(
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