from __future__ import annotations
import time
import cv2
import ctypes
import numpy as np
import numpy.ctypeslib as npct
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple
import platform
import numpy.typing as npt

FFI : bool = False
# FFI : bool = True
CONV_FUNC_NAME = 'conv3d_3p'
DLL_PATH = ''
    
NUM_THREADS = 12

def worker_conv(
    idx: int,
    conv3d_func: ctypes.CDLL,
    all_frames: List[npt.NDArray[np.float32]],
    kernel3d,
    kernel_flat,
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
        p_kern = kernel_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_out  = out_frame.ctypes.data_as(ctypes.c_voidp)

        conv3d_func(
            p_prev,
            p_curr,
            p_next,
            p_kern,
            p_out,
            width,
            height
        )
        return out_frame.astype(np.uint8)
    else:
        stacked = np.stack(frames, axis=0)  # shape (3, h, w, 3)
        out_channels = []
        for c in range(3):
            out = convolve(stacked[:, :, :, c], kernel3d, mode='nearest')
            out_channels.append(out[1])

        out_frame = np.stack(out_channels, axis=2)
        out_frame = np.clip(out_frame * 255, 0, 255).astype(np.uint8)
        return out_frame


def convy(frames : list, k3d) -> bool:
    stacked = np.stack(frames, axis=0)  # shape (3, h, w, 3)
    out_channels = []
    for c in range(3):  # RGB
        out = convolve(stacked[:, :, :, c], k3d, mode='nearest')
        out_channels.append(out[1])  # middle frame result

    out_frame = np.stack(out_channels, axis=2)
    out_frame = np.clip(out_frame * 255, 0, 255).astype(np.uint8)
    writer.write(out_frame)

    # Slide window
    ret, next_frame = cap.read()
    if not ret:
        return False
    frames.pop(0)
    frames.append(next_frame.astype(np.float32) / 255.0)
    return True

def convffi(conv3d_func, frames : list, k3d_flat) -> bool:
    stacked = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)
    out_frame = np.zeros_like(frames[1])
    height, width = stacked.shape[1:3]
    
    conv3d_func(
        npct.as_ctypes(stacked),
        k3d_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        npct.as_ctypes(out_frame),
        width,
        height,
        1
    )

    final_frame = out_frame.astype(np.uint8)
    writer.write(final_frame)

    ret, next_frame = cap.read()
    if not ret:
        return False
    
    frames.pop(0)
    frames.append(next_frame.astype(np.float32))
    return True

def main_real_time():
    global cap
    lib = ctypes.CDLL(DLL_PATH)
    conv3d_func = lib[CONV_FUNC_NAME]
    
    if CONV_FUNC_NAME == 'conv3d':
        conv3d_func.argtypes = [
            ctypes.c_voidp, ctypes.POINTER(ctypes.c_float), ctypes.c_voidp,
            ctypes.c_int, ctypes.c_int
        ]
        conv3d_func.restype = None
    elif CONV_FUNC_NAME == 'conv3d_3p':
        conv3d_func.argtypes = [
            ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp,
            ctypes.POINTER(ctypes.c_float), ctypes.c_voidp,
            ctypes.c_int, ctypes.c_int
        ]
        conv3d_func.restype = None

    kernel3d = np.array([
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0],    [0, 0, 0],    [0, 0, 0]],
        [[1, 2, 1],    [2, 4, 2],    [1, 2, 1]]
    ], dtype=np.float32)
    kernel3d_flat = kernel3d.flatten()

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera opened: {w}x{h}")
    print("Press 'q' to quit.")

    frames = []
    for _ in range(2):
        ret, f = cap.read()
        if not ret: break
        if FFI:
            frames.append(f.astype(np.float32))
        else:
            frames.append(f.astype(np.float32) / 255.0)

    while True:
        ret, f = cap.read()
        f = cv2.flip(f, 1)
        if not ret:
            break
            
        if FFI:
            frames.append(f.astype(np.float32))
        else:
            frames.append(f.astype(np.float32) / 255.0)

        result_frame = worker_conv(
            0,
            conv3d_func,
            frames,
            kernel3d,
            kernel3d_flat,
            w, h
        )
        cv2.imshow("3D Conv Real-Time", result_frame)
        frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main_video():
    global cap, writer
    lib = ctypes.CDLL(DLL_PATH)
    conv3d_func = lib[CONV_FUNC_NAME]
    if CONV_FUNC_NAME == 'conv3d':
        conv3d_func.argtypes = [
            ctypes.c_voidp,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_voidp,
            ctypes.c_int,
            ctypes.c_int]
        conv3d_func.restype = None
    elif CONV_FUNC_NAME == 'conv3d_3p':
        conv3d_func.argtypes = [
            ctypes.c_voidp,
            ctypes.c_voidp,
            ctypes.c_voidp,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_voidp,
            ctypes.c_int,
            ctypes.c_int]
        conv3d_func.restype = None

    kernel3d = np.array([
        [[-1, -2, -1],
        [-2, -4, -2],
        [-1, -2, -1]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]
    ], dtype=np.float32)
    kernel3d_flat = kernel3d.flatten()

    input_path = "./input_videos/sample.mp4"
    output_path = "./output_videos/output_py.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at {input_path}. Please check the path.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))


    start_time = time.time()
    print(f'FFI : {FFI}')
    print(f"Processing video (W:{w}, H:{h}, FPS:{fps})...")


    frames = []
    for _ in range(2):
        ret, f = cap.read()
        if not ret:
            break
        if FFI:
            frames.append(f.astype(np.float32))
        else:
            frames.append(f.astype(np.float32) / 255.0)
    if len(frames) < 2:
        print("Not enough frames to process.")
        return
    while True:

        future_frames: List[npt.NDArray[np.float32]] = []
        for _ in range(NUM_THREADS):
            ret, nf = cap.read()
            if not ret:
                break
            if FFI:
                future_frames.append(nf.astype(np.float32))
            else:
                future_frames.append(nf.astype(np.float32) / 255.0)
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
                    kernel3d,
                    kernel3d_flat,
                    w, h
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
    print(f"✅ Done! Saved to {output_path}")

if __name__ == "__main__":
    IS_WINDOWS = platform.system() == 'Windows'
    if IS_WINDOWS:
        DLL_PATH = './build/conv3d.dll'
        main_real_time()
    else:
        DLL_PATH = './build/libconv3d.so'
        main_video()
