#!/usr/bin/env python
"""
inference_pipeline.py

Flujo completo de:
  1) Inferencia de segmentación 3D U-Net en k-fold (soft-ensemble) → máscaras .npy
  2) Selección de frames con base en área de máscara → clasificación frame-wise (ensemble)
"""

# --- 0) Imports y supresión de warnings ---
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    module="torchvision.io.image"
)

import os
import glob
import argparse
import yaml                       # leer tu config YAML de segmentación
from addict import Dict          # para acceder a conf.train_par.workers como atributo
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Segmentación
import skvideo.io                # para leer vídeos .mp4 como array NumPy
from transforms_seg import Rescale, ToTensor   # tus transforms para segmentación
from model_lightning_seg import MyModel         # tu LightningModule 3D U-Net

# Clasificación
from utils_seg import select_candidate_frames, load_frames_from_video
from utils_clasi import load_model, predict_with_model, summarize_ensemble_predictions
from torchvision import transforms as cls_transforms
from collections import Counter

# Mapeo numérico → texto (debe coincidir con utils_clasi)
LABEL_MAP = {0: "No follow up", 1: "Follow up", 2: "Biopsy"}


# --- 1) Dataset para segmentación ---
class VideoInferenceDataset(Dataset):
    """
    Lee **recursivamente** todos los .mp4 bajo input_dir (incluyendo sub-carpetas)
    y los convierte en volúmenes grises (1, D, H, W) para tu 3D U-Net.
    """
    def __init__(self, input_dir, seg_transform=None):
        pattern = os.path.join(input_dir, "**", "*.mp4")
        self.files = sorted(glob.glob(pattern, recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No se hallaron .mp4 en {input_dir} (buscando recursivamente)")
        self.transform = seg_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        video = skvideo.io.vread(path)      # (D, H, W, 3)
        D, H, W, _ = video.shape

        # rgb→gray volumen (1, D, H, W)
        gray_vol = np.zeros((1, D, H, W), dtype=np.uint8)
        for i in range(D):
            gray = np.dot(video[i], [0.2989, 0.5870, 0.1140])
            gray_vol[0, i] = gray.astype(np.uint8)

        sample = {"image": gray_vol, "filename": os.path.basename(path)}
        if self.transform:
            sample = self.transform(sample)
        return sample


# --- 2) Dataset para clasificación ---
class FrameDataset(Dataset):
    """
    Dataset minimalista que recibe una lista de frames RGB (H, W, C)
    y los transforma para tu CNN 2D.
    """
    def __init__(self, frames, cls_transform=None):
        self.frames = frames
        self.transform = cls_transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        if self.transform:
            img = self.transform(img)
        # devolvemos un label y un patient_id dummy (0, 0)
        # porque predict_with_model ignora esos campos
        return img, 0, 0


# --- 3) Parseo de argumentos ---
def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline: segmentación 3D U-Net + clasificación frame-wise"
    )

    # Segmentación
    p.add_argument("--seg_config",      "-c", required=True,
                   help="YAML config para segmentación")
    p.add_argument("--seg_ckpt_dir",    "-k", required=True,
                   help="Carpeta con checkpoints kfold_*.ckpt")
    p.add_argument("--video_dir",       "-i", required=True,
                   help="Directorio raíz con vídeos .mp4")
    p.add_argument("--out_mask_dir",    "-o", required=True,
                   help="Dónde guardar máscaras .npy")
    p.add_argument("--seg_batch_size",  type=int, default=1,
                   help="Batch size para inferencia de segmentación")

    # Clasificación
    p.add_argument("--cls_ckpt_paths",  nargs="+", required=True,
                   help="Rutas a checkpoints de clasificación (ResNet, etc.)")
    p.add_argument("--n_samples",       type=int, default=5,
                   help="Frames aleatorios + frame de área máxima")
    p.add_argument("--tol",             type=float, default=0.2,
                   help="Tolerancia área (20% respecto al máximo)")
    p.add_argument("--video_ext",       type=str, default=".mp4",
                   help="Extensión de los vídeos para clasificación")
    p.add_argument("--cls_batch_size",  type=int, default=8,
                   help="Batch size para inferencia de clasificación")
    p.add_argument("--output_csv",      type=str,
                   default="ensemble_summary.csv",
                   help="CSV de resumen final")

    return p.parse_args()


# --- 4) Función principal ---
def main():
    args = parse_args()

    # 4.1) ==== SEGMENTACIÓN ====
    # 4.1.1) cargar configuración YAML
    conf = Dict(yaml.safe_load(open(args.seg_config, "r")))

    # 4.1.2) dataset y dataloader
    seg_ds = VideoInferenceDataset(
        args.video_dir,
        seg_transform=__import__("torchvision.transforms.v2", fromlist=["Compose"])\
                          .Compose([Rescale((256,256)), ToTensor()])
    )
    seg_loader = DataLoader(
        seg_ds,
        batch_size = args.seg_batch_size,
        shuffle    = False,
        num_workers= conf.train_par.workers,
        pin_memory = True
    )

    # 4.1.3) cargar todos los folds
    seg_ckpts = sorted(glob.glob(os.path.join(args.seg_ckpt_dir, "kfold_*.ckpt")))
    if not seg_ckpts:
        raise FileNotFoundError(f"No hay ckpts en {args.seg_ckpt_dir}")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_models = []
    for ckpt in seg_ckpts:
        m = MyModel.load_from_checkpoint(
            checkpoint_path=ckpt,
            model_opts=conf.model_opts,
            train_par=conf.train_par,
            strict=False
        ).to(device)
        m.eval()
        seg_models.append(m)

    # 4.1.4) inferencia y soft‐ensemble POR VÍDEO (evita mismatch de dimensiones)
    video_paths = seg_ds.files
    N_videos    = len(video_paths)
    video_probs = {i: [] for i in range(N_videos)}

    with torch.no_grad():
        for m in seg_models:
            for vid_idx, batch in enumerate(seg_loader):
                x     = batch["image"].to(device)    # (1,1,D_i,H,W)
                probs = torch.sigmoid(m(x)).cpu()    # (1,1,D_i,H,W)
                video_probs[vid_idx].append(probs)

    # 4.1.5) promedio y guardado de máscaras .npy
    os.makedirs(args.out_mask_dir, exist_ok=True)
    thr         = conf.train_par.eval_threshold
    video_names = [os.path.basename(p) for p in video_paths]

    for vid_idx, name in enumerate(video_names):
        # apilamos M salidas para el mismo vídeo → (M,1,1,D,H,W)
        stacked = torch.stack(video_probs[vid_idx], dim=0)
        avg     = stacked.mean(dim=0)       # (1,1,D,H,W)
        avg_vol = avg[0,0]                  # (D,H,W)
        mask    = (avg_vol >= thr).numpy().astype(np.uint8)

        out_name = os.path.splitext(name)[0] + "_mask.npy"
        np.save(os.path.join(args.out_mask_dir, out_name), mask)

    print(f"→ Segmentación: guardadas {N_videos} máscaras en {args.out_mask_dir}")


    # 4.2) ==== CLASIFICACIÓN ====
    # 4.2.1) cargar modelos pre-entrenados
    cls_models = []
    for ckpt in args.cls_ckpt_paths:
        model, _ = load_model(ckpt,
                    model_name=os.path.basename(ckpt).split('.')[0])
        cls_models.append(model)

    # 4.2.2) definir transformaciones de imagen
    cls_transform = cls_transforms.Compose([
        cls_transforms.ToPILImage(),
        cls_transforms.Resize((224,224)),
        cls_transforms.ToTensor(),
        cls_transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # 4.2.3) recorrer cada máscara y clasificar sus frames
    all_votes = {}
    for mask_fn in os.listdir(args.out_mask_dir):
        if not mask_fn.endswith("_mask.npy"):
            continue

        # extraer clip ("4.1") y sesión ("1")
        clip      = mask_fn.replace("_mask.npy","")
        session   = clip.split(".")[1]
        mask_path = os.path.join(args.out_mask_dir, mask_fn)

        # buscar recursivamente el vídeo que contenga "4.1" en "004/1/"
        sess_dir = os.path.join(args.video_dir, session)
        pat      = os.path.join(sess_dir, f"*{clip}*{args.video_ext}")
        matches  = glob.glob(pat)
        if not matches:
            raise FileNotFoundError(f"No encontré vídeo con patrón {pat}")
        video_path = matches[0]
        print(f"[Clip {clip}] Vídeo → {video_path}")

        # 4.2.3.1) seleccionar índices de frame con la máscara
        idxs   = select_candidate_frames(
                    mask_path,
                    n_samples=args.n_samples,
                    tol=args.tol
                )
        # 4.2.3.2) leer solo esos frames
        frames = load_frames_from_video(video_path, idxs)

        # 4.2.3.3) dataloader de frames
        ds_cls = FrameDataset(frames, cls_transform)
        dl_cls = DataLoader(ds_cls,
                            batch_size=args.cls_batch_size,
                            shuffle=False,
                            num_workers=2)

        # 4.2.3.4) inferencia ensemble sobre cada frame
        probs_per_model = [
            predict_with_model(m, dl_cls, device)
            for m in cls_models
        ]
        n_frames = probs_per_model[0].shape[0]

        # majority-vote frame-wise
        frame_votes = []
        for i in range(n_frames):
            votes = [LABEL_MAP[np.argmax(p[i])] for p in probs_per_model]
            winner, _ = Counter(votes).most_common(1)[0]
            frame_votes.append(winner)

        all_votes[clip] = frame_votes

    # 4.2.4) resumen y CSV final
    df = summarize_ensemble_predictions(all_votes)
    df.to_csv(args.output_csv, index=False)
    print(f"→ Clasificación: resumen guardado en {args.output_csv}")
    print(df)


if __name__ == "__main__":
    main()
