# inference_seg_folder.py
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from matplotlib import pyplot as plt
import torchio as tio
import glob
import argparse
import yaml
from addict import Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from transforms_seg import ToTensor, Rescale
import skvideo.io
from model_lightning_seg import MyModel
from transforms_seg import ToTensor
import time
import warnings
import gc

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser("3D UNet k-fold folder-based inference")
    p.add_argument(
        "--config",
        "-c",
        default="default_config_train_seg.yaml",
        help="path to your training config (YAML)",
    )
    p.add_argument(
        "--ckpt_dir",
        "-k",
        # required=True,
        help="directory containing your kfold_*.ckpt files",
        default="/Users/emilio/Library/CloudStorage/Box-Box/GitHub/Breast-AI-model/experiment_5_giancarlo",
    )
    p.add_argument(
        "--input_dir",
        "-i",
        # required=True,
        help="folder containing your .mp4 test videos",
        default="/Users/emilio/Library/CloudStorage/Box-Box/GitHub/Breast-AI-model/src/videos",
        # default="/Users/emilio/Library/CloudStorage/Box-Box/GitHub/Breast-AI-model/src/P4/video",
    )
    p.add_argument(
        "--out_dir",
        "-o",
        default="./predictions",
        help="where to save predicted masks (.npy)",
    )

    p.add_argument(
        "--out_overlay_dir",
        "-ov",
        default="./overlays",
        help="where to save overlay videos",
    )
    p.add_argument(
        "--batch_size", "-b", type=int, default=1, help="batch size for inference"
    )
    return p.parse_args()


class VideoInferenceDataset(Dataset):
    """Read every .mp4 under input_dir, convert to [1,D,H,W] gray volumes."""

    def __init__(self, input_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
        if not self.files:
            raise FileNotFoundError(f"No .mp4 files in {input_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        video = skvideo.io.vread(path)  # shape (D,H,W,3)
        # Crop centrado a 912 x 912
        D, H, W, _ = video.shape
        crop_size = 912
        center_x = W // 2
        start_x, end_x = center_x - crop_size // 2, center_x + crop_size // 2

        center_y = H // 2
        start_y, end_y = center_y - crop_size // 2, center_y + crop_size // 2

        video = video[:, start_y:end_y, start_x:end_x, :]  # (D, 912, 912, 3)
        # rgb→gray: dot with lum weights
        vid_rgb2gray = np.zeros(
            (1, video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8
        )
        for i in range(video.shape[0]):
            vid_rgb2gray[0, i, :, :] = np.expand_dims(
                np.dot(video[i], [0.2989, 0.5870, 0.1140]), axis=0
            )

        img = tio.ScalarImage(tensor=vid_rgb2gray)

        img = tio.Resize((128, 128, 128))(img)  # resize to (1,D,H,W)
        vid_rgb2gray = img.data.numpy()  # shape (1,D,H,W)
        sample = {"image": vid_rgb2gray, "filename": os.path.basename(path)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def save_overlay(video_path, out_overlay_dir, mask):
    os.makedirs(out_overlay_dir, exist_ok=True)
    # Leer vídeo
    video = skvideo.io.vread(video_path)  # shape: (D, H, W, 3)

    # Crop centrado a 912 x 912
    D, H, W, _ = video.shape
    crop_size = 912
    center_x = W // 2
    start_x, end_x = center_x - crop_size // 2, center_x + crop_size // 2

    center_y = H // 2
    start_y, end_y = center_y - crop_size // 2, center_y + crop_size // 2

    video = video[:, start_y:end_y, start_x:end_x, :]  # (D, 912, 912, 3)
    # Copia para la salida
    overlay = video.copy().astype(np.uint8)

    D, H, W, _ = overlay.shape
    # Redimensionar la máscara (nota que cv2.resize espera (W, H))
    # mask_resized = np.stack(
    #     [
    #         cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    #         for m in mask
    #     ],
    #     axis=0,
    # )  # shape: (D, H, W)

    mask_resized = tio.LabelMap(tensor=np.expand_dims(mask, axis=0))  # shape (1,D,H,W)

    mask_resized = tio.Resize((D, H, W))(mask_resized)  # resize to (1,D,H,W)
    mask_resized = np.squeeze(mask_resized.data.numpy())  # shape (1,D,H,W)

    # Construir vídeo resultante frame a frame
    out_frames = []
    for i in range(D):
        frame = overlay[i]  # (H, W, 3)
        mask_i = mask_resized[i] * 255  # (H, W), valores 0 o 255

        # Haz un “mapa de color” rojo de la máscara:
        colored_mask = np.zeros_like(frame)
        colored_mask[..., 0] = mask_i  # canal rojo en OpenCV es el índice 2 (BGR)

        # Mezcla: 70% vídeo original, 30% máscara roja
        blended = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        out_frames.append(blended)

    out_array = np.stack(out_frames, axis=0)  # (D, H, W, 3)
    # Escribe el vídeo con la misma resolución y FPS original
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{base}_overlay.mp4"
    skvideo.io.vwrite(os.path.join(out_overlay_dir, out_name), out_array)


def main():
    start_time = time.time()
    args = parse_args()
    conf = Dict(yaml.safe_load(open(args.config, "r")))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_overlay_dir, exist_ok=True)

    # --- device ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- ckpts ---
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "kfold_*.ckpt")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")

    # --- dataset (puedes bajar el tamaño si necesitas memoria) ---
    ds = VideoInferenceDataset(
        args.input_dir,
        transform=transforms.Compose([Rescale(output_size=(128, 128)), ToTensor()]),
    )

    # ---- loop por video ----
    for vid_idx, path in enumerate(ds.files):
        fname = os.path.basename(path)

        # obtén muestra (puede ser dict o tensor dependiendo de transform)
        sample = ds[vid_idx]

        if isinstance(sample, dict):
            x = sample["image"]
        else:
            x = sample  # asumimos tensor

        # asegúrate de tensor float
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # shape ⇒ (1,1,D,H,W)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        x = x.to(device, dtype=torch.float32)

        preds = []

        for c in ckpts:
            m = MyModel.load_from_checkpoint(
                checkpoint_path=c,
                model_opts=conf.model_opts,
                train_par=conf.train_par,
                strict=False,
            ).to(device)

            # usando torch compile

            try:
                if device.type in ["cuda", "cpu"]:
                    m = torch.compile(m)
                elif device.type == "mps":
                    # puedes probar esto si quieres arriesgar
                    m = torch.compile(m, mode="default", fullgraph=False)
            except Exception as e:
                print(f"[!] torch.compile falló: {e}")

            m.eval()

            with torch.no_grad():
                logits = m(x)
                probs = torch.sigmoid(logits)  # (1,1,D,H,W)
                preds.append(probs.cpu())

            # libera modelo antes del siguiente ckpt
            del m
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        # promedio
        avg = torch.stack(preds, dim=0).mean(dim=0)  # (1,1,D,H,W)
        thr = conf.train_par.eval_threshold
        mask = (avg[0, 0] >= thr).numpy().astype(np.uint8)

        # guarda
        out_name = os.path.splitext(fname)[0] + "_masken.npy"
        np.save(os.path.join(args.out_dir, out_name), mask)

        # overlay
        save_overlay(path, args.out_overlay_dir, mask)

        print(f"[✓] {fname} procesado ({vid_idx + 1}/{len(ds)})")

    print(f"\nHecho. Máscaras en: {args.out_dir}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")


if __name__ == "__main__":
    main()
