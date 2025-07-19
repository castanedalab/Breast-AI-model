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


def parse_args():
    p = argparse.ArgumentParser("ONNX ensemble inference for 3D segmentation")
    p.add_argument("--config", "-c", default="default_config_train_seg.yaml")
    p.add_argument(
        "--onnx_dir", "-k", default="./onnx_models", help="Folder with model_fold*.onnx"
    )
    p.add_argument(
        "--input_dir", "-i", default="./videos", help="Folder with test .mp4 videos"
    )
    p.add_argument("--out_dir", "-o", default="./predictions")
    p.add_argument("--out_overlay_dir", "-ov", default="./overlays")
    return p.parse_args()


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
        D, H, W, _ = video.shape
        crop_size = 912
        center_x, center_y = W // 2, H // 2
        video = video[
            :, center_y - 456 : center_y + 456, center_x - 456 : center_x + 456, :
        ]

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
        sample = {"image": vid_rgb2gray, "filename": os.path.basename(path)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def save_overlay(video_path, out_overlay_dir, mask):
    os.makedirs(out_overlay_dir, exist_ok=True)
    video = skvideo.io.vread(video_path)
    D, H, W, _ = video.shape
    center_x, center_y = W // 2, H // 2
    video = video[
        :, center_y - 456 : center_y + 456, center_x - 456 : center_x + 456, :
    ]
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
        print("Using CUDAExecutionProvider for ONNX inference")
        sessions = [
            ort.InferenceSession(p, providers=["CUDAExecutionProvider"])
            for p in onnx_paths
        ]
    else:
        # Use CPUExecutionProvider for non-MPS devices
        print("Using CPUExecutionProvider for ONNX inference")
        sessions = [
            ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            for p in onnx_paths
        ]

    ds = VideoInferenceDataset(
        args.input_dir, transform=transforms.Compose([Rescale((128, 128)), ToTensor()])
    )

    for vid_idx, path in enumerate(ds.files):
        fname = os.path.basename(path)
        sample = ds[vid_idx]
        x = sample["image"] if isinstance(sample, dict) else sample
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        x_numpy = x.numpy()

        preds = []
        for s in sessions:
            out = s.run(["output"], {"input": x_numpy})[0]  # shape (1,1,D,H,W)
            preds.append(out)
            del out
            torch.cuda.empty_cache()
            gc.collect()

        avg = np.mean(preds, axis=0)  # (1,1,D,H,W)

        del preds
        torch.cuda.empty_cache()
        gc.collect()
        thr = conf.train_par.eval_threshold
        mask = (avg[0, 0] >= thr).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + "_masken.npy"
        np.save(os.path.join(args.out_dir, out_name), mask)
        save_overlay(path, args.out_overlay_dir, mask)
        print(f"[✓] {fname} procesado ({vid_idx + 1}/{len(ds)})")

    print(f"\n✅ Inference completa. Resultados en: {args.out_dir}")

    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")


if __name__ == "__main__":
    main()
