# inference_seg_folder.py

import os
import glob
import argparse
import yaml
from addict import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from transforms_seg import ToTensor, Rescale
import skvideo.io
from model_lightning_seg import MyModel
from transforms_seg import ToTensor


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
        required=True,
        help="directory containing your kfold_*.ckpt files",
    )
    p.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="folder containing your .mp4 test videos",
    )
    p.add_argument(
        "--out_dir",
        "-o",
        default="./predictions",
        help="where to save predicted masks (.npy)",
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

        sample = {"image": vid_rgb2gray, "filename": os.path.basename(path)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def main():
    args = parse_args()

    # — load config & grab transforms —
    conf = Dict(yaml.safe_load(open(args.config, "r")))
    # assume you have a function in transforms_seg.py that builds your train/dev transforms

    # — build inference dataset & loader —
    ds = VideoInferenceDataset(
        args.input_dir, transform=transforms.Compose([ToTensor()])
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for c in ckpts:
        m = MyModel.load_from_checkpoint(
            checkpoint_path=c, model_opts=conf.model_opts, train_par=conf.train_par
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
                logits = m(x)
                probs = torch.sigmoid(logits)  # (B,1,D,H,W)
                fold_probs.append(probs.cpu())
            all_probs.append(torch.cat(fold_probs, dim=0))  # (N,1,D,H,W)

    # average
    avg = sum(all_probs) / len(all_probs)  # (N,1,D,H,W)
    thr = conf.train_par.eval_threshold

    # — save per‐video mask —
    os.makedirs(args.out_dir, exist_ok=True)
    names = [os.path.basename(p) for p in ds.files]
    averaged = avg.numpy()
    for i, fname in enumerate(names):
        mask = (averaged[i, 0] >= thr).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + "_mask.npy"
        np.save(os.path.join(args.out_dir, out_name), mask)

    print(f"Saved {len(names)} masks → {args.out_dir}")


if __name__ == "__main__":
    main()
