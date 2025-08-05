import numpy as np
import cv2
import random
from skimage.metrics import structural_similarity as ssim
import os
import skvideo

skvideo.setFFmpegPath("C:/Users/PUCP-ROCHESTER-003/Documents/ffmpeg/bin")
# print("FFmpeg path: {}".format(skvideo.getFFmpegPath()))
# print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))

import skvideo.io


def vread(path):
    """Lectura de video usando OpenCV como np.ndarray [T, H, W, C]"""
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir de BGR a RGB si deseas mantener consistencia con skvideo/imageio
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.stack(frames)


def vwrite(
    fname, videodata, inputdict=None, outputdict=None, backend="ffmpeg", verbosity=0
):
    """
    Guarda un vídeo usando OpenCV, manteniendo la misma firma de skvideo.io.vwrite:
      vwrite(fname, videodata, inputdict=None, outputdict=None, backend='ffmpeg', verbosity=0)

    Parámetros
    ----------
    fname : str
        Ruta de salida (p.ej. os.path.join(out_overlay_dir, out_name))
    videodata : ndarray
        Array con forma (T, H, W, C) o (T, H, W) (uint8).
    inputdict, outputdict, backend, verbosity : ignorados
        Sólo se mantienen para compatibilidad de firma.
    """
    # Asegurarnos de un ndarray y forma 4D
    videodata = np.asarray(videodata)
    if videodata.ndim == 3:
        # (T, H, W) → (T, H, W, 1)
        videodata = videodata[..., np.newaxis]
    T, H, W, C = videodata.shape

    # Elegimos un codec básico según la extensión de salida
    ext = os.path.splitext(fname)[1].lower()
    if ext in [".avi"]:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        # mp4, mov, mkv...
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Frames por segundo fijo (puedes extraerlo de outputdict['-r'] si quieres)
    fps = 30

    # OpenCV espera BGR si es color
    is_color = C == 3
    writer = cv2.VideoWriter(fname, fourcc, fps, (W, H), isColor=is_color)

    for frame in videodata:
        if C == 1:
            # escala de grises
            out_frame = frame[:, :, 0]
        else:
            # RGB → BGR
            out_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(out_frame)

    writer.release()


def select_candidate_frames(
    mask_array, n_samples=5, tol=0.5, video_path=None, crop_coords=None
):
    """
    Selecciona n_samples+1 frames diversos basados en área de máscara y SSIM.

    - mask_array: ndarray (D,H,W) o (1,D,H,W)
    - video_path: ruta al video original (.mp4)
    - crop_coords: (minr, maxr, minc, maxc) usadas durante inferencia
    """
    if mask_array.ndim == 4:
        mask_array = mask_array[0]

    areas = (mask_array > 0).reshape(mask_array.shape[0], -1).sum(1)
    idx_max = int(np.argmax(areas))
    A = areas[idx_max]
    elig = [i for i, a in enumerate(areas) if i != idx_max and a >= (1 - tol) * A]

    if not elig or not video_path or not crop_coords:
        return [idx_max] + random.sample(elig, min(len(elig), n_samples))

    # Leer video y recortar con crop_coords
    video = skvideo.io.vread(video_path)
    minr, maxr, minc, maxc = crop_coords
    video_crop = video[:, minr:maxr, minc:maxc, :]
    gray = np.dot(video_crop[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    ref = gray[idx_max]
    ssim_scores = [(i, ssim(ref, gray[i], data_range=255)) for i in elig]
    ssim_scores.sort(key=lambda x: x[1])  # menor SSIM = más diferente
    selected = [idx for idx, _ in ssim_scores[:n_samples]]

    return [idx_max] + selected


def select_candidate_frames_path(mask_npy_path, n_samples=5, tol=0.2):
    masks = np.load(mask_npy_path)
    areas = (masks > 0).reshape(masks.shape[0], -1).sum(1)
    idx_max = int(np.argmax(areas))
    A = areas[idx_max]
    elig = [i for i, a in enumerate(areas) if i != idx_max and a >= (1 - tol) * A]
    sampled = random.sample(elig, min(len(elig), n_samples))
    return [idx_max] + sampled


def load_frames_from_video(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    out = {}
    for i in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            raise RuntimeError(f"Frame {i}?")
        out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    return [out[i] for i in frame_indices]
