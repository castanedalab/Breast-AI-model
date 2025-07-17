#inference classification folder
# This script performs inference on a set of videos, classifying frames based on pre-trained models 

import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    module="torchvision.io.image"
)

import os
import argparse
from collections import Counter
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils_seg import select_candidate_frames, load_frames_from_video
from utils_clasi import load_model, predict_with_model, summarize_ensemble_predictions

def find_video_for_patient(video_dir, pid, ext):
    pattern = os.path.join(video_dir, str(pid), f"*{ext}")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No se encontró vídeo para paciente {pid} en {pattern}")
    return matches[0]

# Mapeo numérico → texto (coincide con utils_clasi)
LABEL_MAP = {0: "No follow up", 1: "Follow up", 2: "Biopsy"}

class FrameDataset(Dataset):
    """Dataset sencillo para una lista de frames (np.ndarray HxWxC)."""
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        if self.transform:
            img = self.transform(img)
        # devolvemos un label y patient_id dummy para encajar con predict_with_model
        return img, 0, 0


def main(args):
    # 1. Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Cargar clasificadores
    models = []
    for ckpt in args.ckpt_paths:
        model, _ = load_model(ckpt, model_name=os.path.basename(ckpt).split('.')[0])
        models.append(model)

    # 3. Pipeline de transformaciones para clasificación
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    # 4. Por cada paciente (archivo .npy en mask_dir)
    all_votes = {}
    for fn in os.listdir(args.mask_dir):
        # buscamos sólo archivos como "4.1_mask.npy", "4.2_mask.npy", …
        if not fn.endswith("_mask.npy"):
            continue

        # Extraer clip y carpeta-session
        clip    = fn.replace("_mask.npy", "")  # e.g. "4.1"
        session = clip.split(".")[1]           # e.g. "1"
        mask_path = os.path.join(args.mask_dir, fn)

        # Construir ruta al vídeo: <video_dir>/<session>/<clip>.mp4
        video_path = os.path.join(
            args.video_dir,
            session,
            clip + args.video_ext
        )
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No existe vídeo {video_path} para la máscara {fn}")
        print(f"[Clip {clip}] Leyendo vídeo: {video_path}")

        # 4.1 Seleccionar índices de frame con la máscara .npy
        idxs = select_candidate_frames(
            mask_path,
            n_samples=args.n_samples,
            tol=args.tol
        )

        # 4.2 Leer esos frames del vídeo
        frames = load_frames_from_video(video_path, idxs)

        # 4.3 Crear DataLoader
        ds = FrameDataset(frames, transform=transform)
        dl = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=2)

        # 4.4 Predict con cada modelo
        #    predict_with_model devuelve un array (N_frames x N_clases)
        probs_per_model = [
            predict_with_model(m, dl, device)
            for m in models
        ]
        n_frames = probs_per_model[0].shape[0]

        # 4.5 Majority‐vote por frame
        frame_votes = []
        for i in range(n_frames):
            votes = [
                LABEL_MAP[np.argmax(probs[i])]
                for probs in probs_per_model
            ]
            # ganador del frame
            winner, _ = Counter(votes).most_common(1)[0]
            frame_votes.append(winner)

        all_votes[clip] = frame_votes

    # 5. Resumir y guardar
    df = summarize_ensemble_predictions(all_votes)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Resumen guardado en {args.output_csv}")
    print(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pipeline: máscaras .npy → selección de frames → clasificación por ensemble"
    )
    p.add_argument("--mask_dir",    required=True,
                   help="Carpeta con .npy de máscaras")
    p.add_argument("--video_dir",   required=True,
                   help="Carpeta con vídeos (.mp4 u otra extensión)")
    p.add_argument("--video_ext",   default=".mp4",
                   help="Extensión de los vídeos (p. ej. .mp4, .avi)")
    p.add_argument("--ckpt_paths",  nargs="+", required=True,
                   help="Rutas a los checkpoints de los 3 modelos")
    p.add_argument("--output_csv",  default="ensemble_summary.csv",
                   help="Ruta donde guardar el CSV resumen")
    p.add_argument("--n_samples",   type=int, default=5,
                   help="Número de frames aleatorios (además del de área máxima)")
    p.add_argument("--tol",         type=float, default=0.2,
                   help="Tolerancia: % área mínima relativa al máximo")
    p.add_argument("--batch_size",  type=int, default=8,
                   help="Batch size para DataLoader")
    args = p.parse_args()
    main(args)
