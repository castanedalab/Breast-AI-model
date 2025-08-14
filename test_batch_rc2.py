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
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from skimage import measure
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
        default="/data/DATA/private_data/Breast studyrc2",
        help="Carpeta que contiene subcarpetas de pacientes",
    )
    p.add_argument(
        "--out_root",
        "-o",
        default="./predictions_rc2",
        help="Carpeta raíz para máscaras (.npy)",
    )
    p.add_argument(
        "--ov_root",
        "--ov",
        default="./overlays_rc2",
        help="Carpeta raíz para overlays (.mp4)",
    )
    return p.parse_args()



# === Función para remover el marcador brillante tipo "B" ===
def remove_B_marker(
    video, frame_idx=0, roi_frac=(0.15, 0.15), thr_val=200, min_size=100, dilate_rad=8
):
    """
    Detecta y enmascara el marcador brillante (por ejemplo letra "B") en esquina superior izquierda
    """
    D, H, W, C = video.shape
    rh, rw = int(H * roi_frac[0]), int(W * roi_frac[1])
    roi = video[frame_idx][:rh, :rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
    mask_bin = skm.remove_small_objects(bw > 0, min_size=min_size)
    mask_bin = skm.remove_small_holes(mask_bin, area_threshold=min_size)
    labels = measure.label(mask_bin)
    props = measure.regionprops(labels)
    if not props:
        return video.copy()
    best = min(props, key=lambda p: p.centroid[0] ** 2 + p.centroid[1] ** 2)
    blob = labels == best.label
    blob_filled = binary_fill_holes(blob)
    blob_dil = skm.dilation(blob_filled, footprint=disk(dilate_rad))
    full = np.zeros((H, W), bool)
    full[:rh, :rw] = blob_dil
    mask3d = np.repeat(np.repeat(full[None, :, :, None], D, axis=0), 3, axis=3)
    clean = video.copy()
    clean[mask3d] = 0
    return clean


def select_candidate_frames(mask_array, n_samples=5, tol=0.2):
    """
    Solo area y SSIM sobre máscara binaria, sin leer video.
    """
    # mask_array: (D,H,W)
    areas = mask_array.reshape(mask_array.shape[0], -1).sum(1)
    if areas.max() == 0:
        # raise RuntimeError("No hay frames con máscara.")
        return [0] * n_samples, False
    idx_max = int(np.argmax(areas))

    # el resto que tenga algo de máscara
    elig = [
        i
        for i, a in enumerate(areas)
        if i != idx_max and a > 0 and a >= (1 - tol) * areas[idx_max]
    ]

    # si faltan, mete cuantos quieras de los no-vacíos
    if len(elig) < n_samples:
        more = [i for i, a in enumerate(areas) if a > 0 and i != idx_max]
        elig = list(set(elig + more))

    # usa SSIM sobre las máscaras binarias
    from skimage.metrics import structural_similarity as ssim

    ref = mask_array[idx_max].astype(float)
    scores = [(i, ssim(ref, mask_array[i].astype(float), data_range=1.0)) for i in elig]
    scores.sort(key=lambda x: x[1])  # menor → más distinto
    selected = [i for i, _ in scores[:n_samples]]

    return [idx_max] + selected, True


# === Función para detectar zona activa mediante diferencia de frames ===
def process_video_and_get_crop(video, min_obj_size=1000, hole_size=1000):
    """
    Detecta la región activa (ROI) a partir de diferencias entre frames y la recorta
    """
    video = video[:, :, 0:-50, :]  # elimina borde derecho (artefacto)
    D, H, W, _ = video.shape
    ref_idx, idx1, idx2 = random.sample(range(D), 3)
    diff1 = (video[ref_idx].astype(np.float32) - video[idx1].astype(np.float32)).astype(
        np.uint8
    )
    diff2 = (video[ref_idx].astype(np.float32) - video[idx2].astype(np.float32)).astype(
        np.uint8
    )
    gray1 = cv2.cvtColor(diff1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(diff2, cv2.COLOR_RGB2GRAY)
    mask = np.logical_or(gray1 > 0, gray2 > 0)
    mask = skm.binary_erosion(mask)
    mask = skm.remove_small_objects(mask, min_size=min_obj_size)
    mask = skm.remove_small_holes(mask, area_threshold=hole_size)
    mask = skm.opening(mask)
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        raise RuntimeError("No se encontró ningún objeto tras la limpieza.")
    largest = max(props, key=lambda p: p.area)
    minr, minc, maxr, maxc = largest.bbox
    cropped_video = video[:, minr:maxr, minc:maxc, :]
    return cropped_video, (minr, maxr, minc, maxc)


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


# === Dataset de inferencia ===
class VideoInferenceDataset(Dataset):
    def __init__(self, input_dir, use_dynamic_crop=True):
        """
        Dataset que devuelve un volumen preprocesado listo para segmentación ONNX
        """
        self.files = sorted(
            glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No .mp4 files found in {input_dir}")
        self.use_dynamic_crop = use_dynamic_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        video = skvideo.io.vread(path)

        # Recorte: dinámico o centrado
        if self.use_dynamic_crop:
            video, (minr, maxr, minc, maxc) = process_video_and_get_crop(video)
        else:
            D, H, W, _ = video.shape
            crop_size = 912
            cy, cx = H // 2, W // 2
            y1, y2 = cy - crop_size // 2, cy + crop_size // 2
            x1, x2 = cx - crop_size // 2, cx + crop_size // 2
            video = video[:, y1:y2, x1:x2, :]
            minr, maxr, minc, maxc = y1, y2, x1, x2

        # Limpieza de marcador
        video = remove_B_marker(video, thr_val=180, min_size=50, dilate_rad=20)

        # Conversión a escala de grises y resize a (128,128,128)
        _, H, W, _ = video.shape
        # gray_vol = np.zeros((1, D, H, W), dtype=np.uint8)
        # for i in range(D):
        #     gray = np.dot(video[i], [0.2989, 0.5870, 0.1140])
        #     gray_vol[0, i] = gray.astype(np.uint8)
        # coefs float32 para no caer en float64
        coefs = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        # → esto te da float32
        gray = np.tensordot(video, coefs, axes=([3], [0]))

        # normaliza en [0,1] y SIGUE siendo float32
        gray = gray / np.float32(255.0)

        # 3. Construye el ScalarImage YA en float32 normalizado
        img = tio.ScalarImage(tensor=gray[None, ...])  # shape (1,D,H,W), dtype float32
        img = tio.Resize((128, 128, 128))(img)
        volume = img.data.numpy()  # shape (1,128,128,128), dtype float32
        # img = tio.ScalarImage(tensor=gray_vol)

        # img = tio.ScalarImage(tensor=gray)
        # img = tio.Resize((128, 128, 128))(img)
        # volume = img.data.numpy().astype(np.float32)

        return {
            "image": volume,
            "video_clean": video,  # vídeo limpio (sin marcador)
            "filename": os.path.basename(path),
            "crop_coords": (minr, maxr, minc, maxc),
            "video_path": path,
        }



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

        # entra en "VSI 8 Clips"
        clips_dir = os.path.join(patient_dir, "8 VSI Clips")
        if not os.path.isdir(clips_dir):
            print(f"⚠️  Skipping '{patient_id}': no existe '{clips_dir}'")
            continue

        # prepara carpetas de salida (siguen usando patient_id)
        out_dir = os.path.join(args.out_root, patient_id)
        ov_dir = os.path.join(args.ov_root, patient_id)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ov_dir, exist_ok=True)

        # dataset sobre la carpeta de clips
        ds = VideoInferenceDataset(
            clips_dir)
        
        print(f"\n--- Procesando {patient_id}: {len(ds)} videos en 'VSI 8 Clips' ---")

        for vid_idx, path in enumerate(ds.files):
            sample = ds[vid_idx]
            x = sample["image"]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if x.ndim == 4:
                x = x.unsqueeze(0)
            x_np = x.numpy()

            # inference ensemble ONNX
            preds = []
            for mpath in onnx_paths:
                sess = ort.InferenceSession(mpath, providers=providers)
                out = sess.run(["output"], {"input": x_np})[0]
                preds.append(out)
                del sess, out
                gc.collect()

            avg = np.mean(preds, axis=0)[0, 0]
            mask = (avg >= conf.train_par.eval_threshold).astype(np.uint8)

            # guarda máscara
            mask_name = os.path.splitext(os.path.basename(path))[0] + "_masken.npy"
            np.save(os.path.join(out_dir, mask_name), mask)

            # guarda overlay de tres paneles
            save_three_panel(path, ov_dir, mask)

            print(f"[{vid_idx + 1}/{len(ds)}] {os.path.basename(path)} ✓")

    print("\n✅ ¡Batch completo!")


if __name__ == "__main__":
    main()
