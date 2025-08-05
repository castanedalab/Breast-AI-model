import os
import glob
import argparse
import yaml
from addict import Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torchio as tio
import skvideo.io
import onnxruntime as ort
import gc
import time
from transforms_seg import ToTensor, Rescale
import time
import skimage.morphology as skm
from skimage.morphology import disk, binary_dilation
from scipy.ndimage import binary_fill_holes
from skimage import measure
import random

random.seed(42)  # For reproducibility


def parse_args():
    p = argparse.ArgumentParser("ONNX ensemble inference for 3D segmentation")
    p.add_argument("--config", "-c", default="default_config_train_seg.yaml")
    p.add_argument(
        "--onnx_dir",
        "-k",
        default="./onnx_models",
        help="Folder with model_fold*.onnx",
    )
    p.add_argument(
        "--input_dir", "-i", default="./videos", help="Folder with .mp4 videos"
    )
    p.add_argument("--out_dir", "-o", default="./predictions")
    p.add_argument("--out_overlay_dir", "-ov", default="./overlays_crop")
    return p.parse_args()


def remove_B_marker(
    video, frame_idx=0, roi_frac=(0.15, 0.15), thr_val=200, min_size=100, dilate_rad=8
):
    D, H, W, C = video.shape

    # 1) ROI top‑left
    rh, rw = int(H * roi_frac[0]), int(W * roi_frac[1])
    frame = video[frame_idx]
    roi = frame[:rh, :rw]

    # 2) Binarizo brillo alto
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)

    # 3) Limpio ruido
    mask_bin = skm.remove_small_objects(bw > 0, min_size=min_size)
    mask_bin = skm.remove_small_holes(mask_bin, area_threshold=min_size)

    # 4) Etiqueto y elijo blob más cercano a (0,0)
    labels = measure.label(mask_bin)
    props = measure.regionprops(labels)
    if not props:
        return video.copy()
    best = min(props, key=lambda p: p.centroid[0] ** 2 + p.centroid[1] ** 2)
    blob = labels == best.label

    # --- NUEVO PASO: relleno los “agujeros” interiores ---
    blob_filled = binary_fill_holes(blob)

    # 5) Dilato un poco para sobrecubrir el círculo
    blob_dil = skm.dilation(blob_filled, footprint=disk(dilate_rad))

    # 6) Construyo máscara full-frame y expando a 3D
    full = np.zeros((H, W), bool)
    full[:rh, :rw] = blob_dil
    mask3d = full[None, :, :, None]
    mask3d = np.repeat(mask3d, D, axis=0)
    mask3d = np.repeat(mask3d, 3, axis=3)

    # 7) Pongo a cero esa región en todo el vídeo
    clean = video.copy()
    clean[mask3d] = 0
    return clean


def process_video_and_get_crop(
    video, min_obj_size=1000, hole_size=1000, opening_radius=20
):
    """
    1) Lee el video
    2) Toma dos restas aleatorias de frames
    3) Une, limpia, mantiene el objeto más grande
    4) Devuelve la máscara limpia, la caja de recorte y el frame recortado
    """
    # 1) Abre el video
    video = video[:, :, 0:-50, :]
    D, H, W, C = video.shape

    # 2) Escoge tres índices aleatorios y crea dos máscaras por resta
    ref_idx, idx1, idx2 = random.sample(range(D), 3)
    # convierto a float32 para la resta
    f_ref = video[ref_idx].astype(np.float32)
    f1 = video[idx1].astype(np.float32)
    f2 = video[idx2].astype(np.float32)
    diff1 = (f_ref - f1).astype(np.uint8)
    diff2 = (f_ref - f2).astype(np.uint8)

    # 3) Pasa a gris y binariza
    gray1 = cv2.cvtColor(diff1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(diff2, cv2.COLOR_RGB2GRAY)
    m1 = gray1 > 0
    m2 = gray2 > 0

    # 4) Combina máscaras y limpia pequeños objetos y agujeros
    mask = np.logical_or(m1, m2)
    # mask = np.logical_or(mask, m3)
    # erode para eliminar ruido
    mask = skm.binary_erosion(mask)
    mask = skm.remove_small_objects(mask, min_size=min_obj_size)
    mask = skm.remove_small_holes(mask, area_threshold=hole_size)
    mask = skm.opening(mask)

    # 5) Etiqueta componentes y elige el más grande
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        raise RuntimeError("No se encontró ningún objeto tras la limpieza.")

    largest = max(props, key=lambda p: p.area)
    minr, minc, maxr, maxc = largest.bbox

    # 6) Recorta la máscara y el frame de referencia
    cropped_video = video[:, minr:maxr, minc:maxc, :]
    return cropped_video, (minr, maxr, minc, maxc)


def cropper(video, crop_size=912):
    D, H, W, _ = video.shape

    # Calcula el semilado del recorte
    crop_size = 912
    half = crop_size // 2

    # Centros
    cy, cx = H // 2, W // 2

    # Candidatos de inicio y fin
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    # Ajusta para no salirse de los bordes
    if y1 < 0:
        y1, y2 = 0, crop_size
    if x1 < 0:
        x1, x2 = 0, crop_size
    if y2 > H:
        y2, y1 = H, H - crop_size
    if x2 > W:
        x2, x1 = W, W - crop_size

    # Finalmente recorta
    video = video[:, y1 : y1 + crop_size, x1 : x1 + crop_size, :]
    return video


class VideoInferenceDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
        if not self.files:
            raise FileNotFoundError(f"No .mp4 files in {input_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        video = skvideo.io.vread(path)

        video, (minr, maxr, minc, maxc) = process_video_and_get_crop(video)
        video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

        # video = cropper(video, crop_size=912)

        vid_rgb2gray = np.zeros(
            (1, video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8
        )
        for i in range(video.shape[0]):
            vid_rgb2gray[0, i, :, :] = np.expand_dims(
                np.dot(video[i], [0.2989, 0.5870, 0.1140]), axis=0
            )

        img = tio.ScalarImage(tensor=vid_rgb2gray)
        img = tio.Resize((128, 128, 128))(img)
        vid_rgb2gray = img.data.numpy()
        # minr, maxr, minc, maxc = 0, 128, 0, 128  # Placeholder for crop coords
        crop_coords = (minr, maxr, minc, maxc)
        # sample = {
        #     "image": vid_rgb2gray,
        #     "filename": os.path.basename(path),
        #     "crop_coords": (minr, maxr, minc, maxc),
        # }
        return {
            "image": vid_rgb2gray,  # numpy array
            "filename": os.path.basename(path),
            "crop_coords": crop_coords,
        }

        # if self.transform:
        #     sample = self.transform(sample)
        # return sample


def save_overlay(video_path, out_overlay_dir, mask):
    os.makedirs(out_overlay_dir, exist_ok=True)
    video = skvideo.io.vread(video_path)
    video = video[:, :, 0:-50, :]
    video, (minr, maxr, minc, maxc) = process_video_and_get_crop(video)
    video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

    # D, H, W, _ = video.shape
    # center_x, center_y = W // 2, H // 2
    # video = video[
    #     :, center_y - 456 : center_y + 456, center_x - 456 : center_x + 456, :
    # ]
    overlay = video.copy().astype(np.uint8)
    D, H, W, _ = overlay.shape

    mask_resized = tio.LabelMap(tensor=np.expand_dims(mask, axis=0))
    mask_resized = tio.Resize((D, H, W))(mask_resized)
    mask_resized = np.squeeze(mask_resized.data.numpy())

    out_frames = []
    for i in range(D):
        frame = overlay[i]
        mask_i = mask_resized[i] * 255
        colored_mask = np.zeros_like(frame)
        colored_mask[..., 0] = mask_i
        blended = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        out_frames.append(blended)

    out_array = np.stack(out_frames, axis=0)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{base}_overlay.mp4"
    skvideo.io.vwrite(os.path.join(out_overlay_dir, out_name), out_array)


def save_three_panel(video_path, out_overlay_dir, mask, crop_coords):
    os.makedirs(out_overlay_dir, exist_ok=True)
    # read & crop video to (D, H, W, C)
    video = skvideo.io.vread(video_path)
    video, _ = process_video_and_get_crop(video)
    video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

    # video = video[:, :, 0:-50, :]
    # video = video[
    #     :, crop_coords[0] : crop_coords[1], crop_coords[2] : crop_coords[3], :
    # ]

    # video = cropper(video, crop_size=912)
    # video = process_video_and_get_crop(video)
    D, H, W, C = video.shape

    # prepare mask: (D, H, W), values 0/1
    mask_map = tio.LabelMap(tensor=np.expand_dims(mask, axis=0))
    mask_map = tio.Resize((D, H, W))(mask_map)
    mask_resized = np.squeeze(mask_map.data.numpy())  # shape (D,H,W)

    out_frames = []
    for i in range(D):
        orig = video[i].astype(np.uint8)  # (H,W,3)

        # panel 2: mask in white on black, 3‑channel
        m = (mask_resized[i] * 255).astype(np.uint8)  # (H,W)
        mask_rgb = np.stack([m, m, m], axis=-1)  # (H,W,3)

        # panel 3: overlay in red over orig
        colored_mask = np.zeros_like(orig)
        colored_mask[..., 0] = m  # red channel
        overlay = cv2.addWeighted(orig, 0.7, colored_mask, 0.3, 0)

        # concatenate panels horizontally
        three_panel = np.concatenate([orig, mask_rgb, overlay], axis=1)
        out_frames.append(three_panel)

    out_array = np.stack(out_frames, axis=0)  # (D, H, 3*W, 3)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{base}_overlay.mp4"
    skvideo.io.vwrite(os.path.join(out_overlay_dir, out_name), out_array)


def main():
    # timer
    start_time = time.time()
    args = parse_args()
    conf = Dict(yaml.safe_load(open(args.config, "r")))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_overlay_dir, exist_ok=True)

    # Load ONNX models
    onnx_paths = sorted(glob.glob(os.path.join(args.onnx_dir, "model_fold*.onnx")))
    if not onnx_paths:
        raise RuntimeError(f"No ONNX models found in {args.onnx_dir}")
    # if cuda is available, use CUDAExecutionProvider
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
        print("Using CUDAExecutionProvider for ONNX inference")
    else:
        providers = ["CPUExecutionProvider"]
        # Use CPUExecutionProvider for non-MPS devices
        print("Using CPUExecutionProvider for ONNX inference")

    ds = VideoInferenceDataset(
        args.input_dir, transform=transforms.Compose([ToTensor()])
    )

    onnx_paths = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))

    for vid_idx, path in enumerate(ds.files):
        fname = os.path.basename(path)
        sample = ds[vid_idx]
        x = sample["image"] if isinstance(sample, dict) else sample
        (minr, maxr, minc, maxc) = sample["crop_coords"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x).float()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        x_numpy = x.numpy()

        preds = []
        for model_path in onnx_paths:
            s = ort.InferenceSession(model_path, providers=providers)
            out = s.run(["output"], {"input": x_numpy})[0]
            preds.append(out)
            del s, out
            torch.cuda.empty_cache()
            gc.collect()

        avg = np.mean(preds, axis=0)
        del preds
        torch.cuda.empty_cache()

        thr = conf.train_par.eval_threshold
        mask = (avg[0, 0] >= thr).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + "_masken.npy"
        np.save(os.path.join(args.out_dir, out_name), mask)
        save_three_panel(path, args.out_overlay_dir, mask, (minr, maxr, minc, maxc))
        print(f"[✓] {fname} procesado ({vid_idx + 1}/{len(ds)})")
    print(f"\n✅ Inference completa. Resultados en: {args.out_dir}")

    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")


if __name__ == "__main__":
    main()
