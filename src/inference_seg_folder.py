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

import warnings

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
        # rgb→gray: dot with lum weights
        vid_rgb2gray = np.zeros(
            (1, video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8
        )
        for i in range(video.shape[0]):
            vid_rgb2gray[0, i, :, :] = np.expand_dims(
                np.dot(video[i], [0.2989, 0.5870, 0.1140]), axis=0
            )
        print(vid_rgb2gray.shape)

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
    args = parse_args()

    # — load config & grab transforms —
    conf = Dict(yaml.safe_load(open(args.config, "r")))
    # assume you have a function in transforms_seg.py that builds your train/dev transforms

    # — build inference dataset & loader —
    ds = VideoInferenceDataset(
        args.input_dir,
        transform=transforms.Compose([Rescale(output_size=(128, 128)), ToTensor()]),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=conf.train_par.workers,
        pin_memory=True,
    )

    # — find ckpts & load models —
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "kfold_*.ckpt")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    models = []
    for c in ckpts:
        m = MyModel.load_from_checkpoint(
            checkpoint_path=c,
            model_opts=conf.model_opts,
            train_par=conf.train_par,
            strict=False,
        ).to(device)
        m.eval()
        models.append(m)

    # — inference & soft‐ensemble —
    all_probs = []  # will be list of tensors [N,1,D,H,W] per fold
    with torch.no_grad():
        for m in models:
            fold_probs = []
            for batch in loader:
                x = batch["image"].to(device)  # shape (B,1,D,H,W)
                print(x.shape)
                logits = m(x)
                probs = torch.sigmoid(logits)  # (B,1,D,H,W)
                fold_probs.append(probs.cpu())
            all_probs.append(torch.cat(fold_probs, dim=0))  # (N,1,D,H,W)

    # save all_probs as npy array
    # np.save(os.path.join(args.out_dir, "all_probs.npy"), all_probs)

    # average
    avg = sum(all_probs) / len(all_probs)  # (N,1,D,H,W)
    thr = conf.train_par.eval_threshold
    # — save per‐video mask —
    os.makedirs(args.out_dir, exist_ok=True)
    names = [os.path.basename(p) for p in ds.files]
    averaged = avg.numpy()
    for i, fname in enumerate(names):
        mask = (averaged[i, 0] >= thr).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + "_masken.npy"
        np.save(os.path.join(args.out_dir, out_name), mask)
        video_path = os.path.join(args.input_dir, fname)
        save_overlay(video_path, args.out_overlay_dir, mask)

        # save_overlay(args.input_dir, names[i], args.out_overlay_dir, mask)

    print(f"Saved {len(names)} masks → {args.out_dir}")


if __name__ == "__main__":
    main()
