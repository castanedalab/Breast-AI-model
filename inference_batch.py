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


def parse_args():
    p = argparse.ArgumentParser(
        "ONNX ensemble inference for 3D segmentation (batch over folders)"
    )
    p.add_argument("--config", "-c", default="default_config_train_seg.yaml")
    p.add_argument(
        "--input_root",
        "-i",
        default="./nomass",
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


def cropper(video, crop_size=912):
    D, H, W, _ = video.shape
    half = crop_size // 2
    cy, cx = H // 2, W // 2
    y1, x1 = max(0, cy - half), max(0, cx - half)
    y2, x2 = min(H, cy + half), min(W, cx + half)
    # ajusta si sobresale
    if y2 - y1 < crop_size:
        y1, y2 = H - crop_size, H
    if x2 - x1 < crop_size:
        x1, x2 = W - crop_size, W
    return video[:, y1 : y1 + crop_size, x1 : x1 + crop_size, :]


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
        video = cropper(video, 912)
        # rgb→gray
        vid = np.zeros((1, *video.shape[:3]), dtype=np.uint8)
        for i in range(video.shape[0]):
            vid[0, i] = (video[i] @ [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        img = tio.ScalarImage(tensor=vid)
        img = tio.Resize((128, 128, 128))(img)
        arr = img.data.numpy()
        sample = {"image": arr, "filename": os.path.basename(path)}
        return self.transform(sample) if self.transform else sample


def save_overlay(video_path, out_overlay_dir, mask):
    os.makedirs(out_overlay_dir, exist_ok=True)
    video = skvideo.io.vread(video_path)
    video = cropper(video, 912)
    D, H, W, _ = video.shape
    mask3d = tio.LabelMap(tensor=mask[None])
    mask3d = tio.Resize((D, H, W))(mask3d).data.numpy().squeeze()
    frames = []
    for i in range(D):
        frm = video[i].astype(np.uint8)
        m = (mask3d[i] * 255).astype(np.uint8)
        cm = np.zeros_like(frm)
        cm[..., 0] = m
        frames.append(cv2.addWeighted(frm, 0.7, cm, 0.3, 0))
    out_name = os.path.splitext(os.path.basename(video_path))[0] + "_overlay.mp4"
    skvideo.io.vwrite(os.path.join(out_overlay_dir, out_name), np.stack(frames, 0))


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
            save_overlay(path, ov_dir, mask)
            print(f"[{vid_idx + 1}/{len(ds)}] {os.path.basename(path)} ✓")
    print("\n✅ ¡Batch completo!")


if __name__ == "__main__":
    main()
