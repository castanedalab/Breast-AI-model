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
import skimage.morphology as skm
from skimage import measure
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk
import random
from transforms_seg import ToTensor, Rescale

random.seed(42)  # For reproducibility


def parse_args():
    p = argparse.ArgumentParser(
        "ONNX ensemble inference for 3D segmentation (batch over folders)"
    )
    p.add_argument("--config", "-c", default="default_config_train_seg.yaml")
    p.add_argument(
        "--input_root",
        "-i",
        default="/data/DATA/private_data/Breast AI study mp4",
        help="Carpeta que contiene subcarpetas de pacientes",
    )
    p.add_argument(
        "--out_root",
        "-o",
        default="./predictions",
        help="Carpeta raíz para máscaras (.npy)",
    )
    p.add_argument(
        "--ov_root",
        "--ov",
        default="./overlays",
        help="Carpeta raíz para overlays (.mp4)",
    )
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
    return cropped_video


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
        video = process_video_and_get_crop(video)
        video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

        # rgb→gray
        vid = np.zeros((1, *video.shape[:3]), dtype=np.uint8)
        for i in range(video.shape[0]):
            vid[0, i] = (video[i] @ [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        img = tio.ScalarImage(tensor=vid)
        img = tio.Resize((128, 128, 128))(img)
        arr = img.data.numpy()
        sample = {"image": arr, "filename": os.path.basename(path)}
        return self.transform(sample) if self.transform else sample


def save_three_panel(video_path, out_overlay_dir, mask):
    os.makedirs(out_overlay_dir, exist_ok=True)
    # read & crop video to (D, H, W, C)
    video = skvideo.io.vread(video_path)
    video = process_video_and_get_crop(video)
    video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

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
    args = parse_args()
    conf = Dict(yaml.safe_load(open(args.config)))
    # carga sesiones ONNX UNA sola vez
    onnx_paths = sorted(glob.glob(os.path.join(os.getcwd(), "onnx_models", "*.onnx")))
    providers = (
        ["CUDAExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    print("Providers:", providers)

    # itera sobre cada carpeta de paciente
    for patient_dir in sorted(glob.glob(os.path.join(args.input_root, "*"))):
        if not os.path.isdir(patient_dir):
            continue
        patient_id = os.path.basename(patient_dir)
        out_dir = os.path.join(args.out_root, patient_id)
        ov_dir = os.path.join(args.ov_root, patient_id)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ov_dir, exist_ok=True)

        ds = VideoInferenceDataset(
            patient_dir, transform=transforms.Compose([ToTensor()])
        )
        print(f"\n--- Procesando {patient_id}: {len(ds)} videos ---")
        for vid_idx, path in enumerate(ds.files):
            sample = ds[vid_idx]
            x = sample["image"]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if x.ndim == 4:
                x = x.unsqueeze(0)
            x_np = x.numpy()

            preds = []
            for mpath in onnx_paths:
                sess = ort.InferenceSession(mpath, providers=providers)
                out = sess.run(["output"], {"input": x_np})[0]
                preds.append(out)
                del sess, out
                gc.collect()

            avg = np.mean(preds, axis=0)[0, 0]
            mask = (avg >= conf.train_par.eval_threshold).astype(np.uint8)
            np.save(
                os.path.join(
                    out_dir, os.path.splitext(os.path.basename(path))[0] + "_masken.npy"
                ),
                mask,
            )
            save_three_panel(path, ov_dir, mask)
            print(f"[{vid_idx + 1}/{len(ds)}] {os.path.basename(path)} ✓")
    print("\n✅ ¡Batch completo!")


if __name__ == "__main__":
    main()
