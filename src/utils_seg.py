import numpy as np
import cv2
import random

def select_candidate_frames(mask_npy_path, n_samples=5, tol=0.2):
    masks = np.load(mask_npy_path)
    areas = (masks>0).reshape(masks.shape[0],-1).sum(1)
    idx_max = int(np.argmax(areas)); A = areas[idx_max]
    elig = [i for i,a in enumerate(areas) if i!=idx_max and a>= (1-tol)*A]
    sampled = random.sample(elig, min(len(elig), n_samples))
    return [idx_max] + sampled

def load_frames_from_video(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    out = {}
    for i in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret: raise RuntimeError(f"Frame {i}?")
        out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    return [out[i] for i in frame_indices]
