#!/usr/bin/env python
"""
Pipeline de inferencia completo:
  1) Segmentación 3D usando modelos ONNX (ensemble k-fold)
  2) Clasificación frame-wise usando modelos PyTorch (ResNet, etc.)

Incluye:
- Preprocesamiento completo del video: limpieza de marcador, recorte, resize
- Overlay opcional
- Exportación opcional de máscaras .npy
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    module="torchvision.io.image",
)

# === Librerías estándar y de procesamiento ===
import gc
import os
import glob
import argparse
import yaml
import numpy as np
import torch
import onnxruntime as ort
import cv2
import torchio as tio

# import skvideo.io
import random
import skimage.morphology as skm
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from skimage import measure
from addict import Dict
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as cls_transforms
import matplotlib.pyplot as plt

# === Utilidades auxiliares de segmentación y clasificación ===
# from utils_seg import select_candidate_frames, load_frames_from_video  # , vread, vwrite
from utils_clasi import (
    load_model,
    predict_with_model,
    summarize_ensemble_predictions,
    load_model_onnx,
    predict_with_model_onnx,
)
import skvideo

skvideo.setFFmpegPath("./ffmpeg/bin")
# print("FFmpeg path: {}".format(skvideo.getFFmpegPath()))
# print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))

import skvideo.io

# Mapa numérico a etiquetas para clasificación
LABEL_MAP = {0: "No follow up", 1: "Follow up", 2: "Refer to specialist"}


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


# === Parseo de argumentos ===
def parse_args():
    p = argparse.ArgumentParser("Segmentación ONNX + Clasificación")
    # Argumentos de segmentación
    # p.add_argument(
    #     "--seg_config", "-c", required=True
    # )  # YAML con parámetros de evaluación (umbral, etc.)
    # p.add_argument("--seg_onnx_dir", "-k", required=True)  # Carpeta con modelos ONNX
    p.add_argument("--video_dir", "-i", required=True)  # Carpeta raíz con videos .mp4
    p.add_argument(
        "--out_dir", "-o", required=True
    )  # Carpeta para guardar las máscaras .npy (opcional)
    p.add_argument(
        "--save_overlay", action="store_true"
    )  # Flag para guardar overlay como video

    # Argumentos de clasificación
    # p.add_argument(
    #     "--cls_onnx_dir", required=True
    # )  # Modelos PyTorch entrenados (ResNet, etc.)
    p.add_argument(
        "--n_samples", type=int, default=5
    )  # Nº de frames a clasificar (aleatorios + max área)
    p.add_argument("--tol", type=float, default=0.5)  # Tolerancia sobre área máxima
    p.add_argument(
        "--video_ext", type=str, default=".mp4"
    )  # Extensión de los videos para búsqueda
    p.add_argument(
        "--cls_batch_size", type=int, default=1
    )  # Batch size de inferencia clasificación
    p.add_argument("--output_csv", type=str, default="result.csv")  # Output final
    return p.parse_args()


def save_classification_frames(frames, output_dir, clipname):
    """
    Guarda los frames usados para clasificación como imágenes PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for i, frame in enumerate(frames):
        filename = f"{clipname}_f{i}.png"
        path = os.path.join(output_dir, filename)
        plt.imsave(path, frame)
        saved_paths.append(os.path.join("frames", filename))  # ruta relativa
    return saved_paths


# === Función principal ===
def main():
    args = parse_args()

    # Inicialización de modelos ONNX
    # providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    providers = ["CPUExecutionProvider"]
    onnx_paths = sorted(glob.glob(os.path.join("./segmentacion_onnx", "*.onnx")))
    print(f"Modelos ONNX encontrados: {len(onnx_paths)}")
    onnx_sessions = [ort.InferenceSession(p, providers=providers) for p in onnx_paths]
    onnx_paths = sorted(glob.glob(os.path.join("./clasificacion_onnx", "*.onnx")))
    cls_models = [ort.InferenceSession(p, providers=providers) for p in onnx_paths]

    # Dataset de videos ya preprocesado
    dataset = VideoInferenceDataset(args.video_dir)

    # Diccionario para acumular predicciones por video
    all_votes = {}

    # Asegura que exista el output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # === Bucle de inferencia sobre todos los videos ===
    for i in range(len(dataset)):
        print(f"Procesando video {i + 1}/{len(dataset)}...")
        sample = dataset[i]
        x = sample["image"][None, ...]  # Añade batch dim: (1, 1, D, H, W)
        fname = sample["filename"]
        video_path = sample["video_path"]
        print(f"Procesando video: {video_path}")
        crop_coords = sample["crop_coords"]

        # Ensemble de predicciones ONNX
        preds = []
        for sess in onnx_sessions:
            out = sess.run(["output"], {"input": x})[0]
            preds.append(out)

        # Promedio soft del ensemble
        avg = np.mean(preds, axis=0)  # (1, 1, D, H, W)

        # === Guardado opcional de máscara ===
        # out_name = os.path.splitext(fname)[0] + "_mask.npy"
        # np.save(os.path.join(args.out_mask_dir, out_name), mask)

        # === Guardado opcional de overlay ===
        # if args.save_overlay:
        #     from inference_seg_onnx import (
        #         save_three_panel,
        #     )  # Reutiliza función existente
        if args.save_overlay:
            mask = (avg[0, 0] >= 0.5).astype(np.uint8)
            save_three_panel(video_path, args.out_dir + "/videos", mask, crop_coords)
        else:
            mask = avg[0, 0] >= 0.5
        # === CLASIFICACIÓN ===
        clip = os.path.splitext(fname)[0]

        video_clean = sample["video_clean"]
        # Selección de frames representativos usando la máscara
        # idxs = select_candidate_frames(
        #     mask,
        #     n_samples=args.n_samples,
        #     tol=args.tol,
        #     video_path=video_path,
        #     crop_coords=crop_coords,
        # )
        idxs, detected = select_candidate_frames(
            mask, n_samples=args.n_samples, tol=args.tol
        )

        idxs_orig = [int(idx * video_clean.shape[0] / mask.shape[0]) for idx in idxs]
        frames = [video_clean[i] for i in idxs_orig]
        # frames = load_frames_from_video(video_path, idxs)

        # Guardar frames usados para clasificación
        frames_dir = os.path.join(args.out_dir, "frames")
        frame_paths = save_classification_frames(frames, frames_dir, clip)
        if not detected:
            print(f"⚠️ No se detectaron frames válidos para {clip}. Usando frame 0.")
            frame_paths = [os.path.join("frames", f"{clip}_f0.png")]
            frame_votes = ["No follow up"] * len(frames)  # Predicción por defecto
            del sample, x, mask, frames

        else:
            # Dataset + DataLoader para esos frames
            ds_cls = FrameDataset(
                frames,
                cls_transforms.Compose(
                    [
                        cls_transforms.ToPILImage(),
                        cls_transforms.Resize((224, 224)),
                        cls_transforms.ToTensor(),
                        cls_transforms.Normalize([0.5] * 3, [0.5] * 3),
                    ]
                ),
            )
            dl_cls = DataLoader(
                ds_cls, batch_size=args.cls_batch_size, shuffle=False, num_workers=2
            )

            # Carga de modelos clasificadores
            # Inferencia con cada modelo → softmax por frame
            probs = [predict_with_model_onnx(m, dl_cls) for m in cls_models]

            # Voting por frame (majority vote)
            frame_votes = []
            for f in range(len(frames)):
                votes = [LABEL_MAP[np.argmax(p[f])] for p in probs]
                winner, _ = Counter(votes).most_common(1)[0]
                frame_votes.append(winner)
            del sample, x, mask, frames, dl_cls, probs
        all_votes[clip] = {"votes": frame_votes, "frame_paths": frame_paths}

        gc.collect()

    # === Exportar CSV final con resumen ===
    df = summarize_ensemble_predictions(all_votes)
    print("Predicciones por video:")
    print(df)
    df.to_csv(os.path.join(args.out_dir, args.output_csv), index=False)

    # === POSTPROCESAMIENTO DEL CSV ===

    # 1. Añadir columna 'representative_path' con ruta relativa (una imagen por clip)
    df["representative_path"] = {
        "path": frame_paths[0]
    }  # Debido a que el primer frame es el index mas grande

    # 2. Convertir 'final_label' a formato JSON como dict string
    df["json_label"] = df["final_label"].apply(lambda x: '{ "result": "' + x + '" }')

    # Guardar CSV nuevamente con nuevas columnas
    df.to_csv(os.path.join(args.out_dir, args.output_csv), index=False)
    print(f"✅ CSV actualizado con columnas 'image_path' y 'json_label'")


# === Dataset para clasificación (frames individuales) ===
class FrameDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.frames[idx]
        if self.transform:
            x = self.transform(x)
        return x, 0, 0  # Dummy label y patient_id (no se usan)


if __name__ == "__main__":
    main()
