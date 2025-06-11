import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
import skvideo.io
import cv2

class WSIDataset(Dataset):
    def __init__(self, meta_data, root_dir,  cache_data=False, transform=None):
        df = pd.read_csv(meta_data)
        self.wsi_list = df.wsi.to_list()
        
        self.root_dir = root_dir
        self.transform = transform
        self.cache_data = cache_data
        self.videos_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'video','*.avi')) for i in range(len(self.wsi_list))])
        self.gt_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'mask', '*.avi')) for i in range(len(self.wsi_list))])

    def __len__(self):
        return len(self.videos_path)

    def __getitem__(self, idx):
        '''
        - images come from a single WSI.
        - mask --> 1 for plaques, 0 for background.
        '''
        if self.cache_data is True:
            print("Loading from cache...")
            image = self.dataset_imgs[idx]
        else:
            video = skvideo.io.vread(self.videos_path[idx])
            gt_img = skvideo.io.vread(self.gt_path[idx])

        vid_rgb2gray = np.zeros((1, video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8)
        for i in range(video.shape[0]):
            vid_rgb2gray[0, i, :, :] = np.expand_dims(np.dot(video[i], [0.2989, 0.5870, 0.1140]), axis=0)
        
        gt_rgb2gray = np.zeros((1, gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]), dtype=np.uint8)
        # print("Original mask",gt_img.min(), gt_img.max())
        for i in range(gt_img.shape[0]):
            gt_rgb2gray[0, i, :, :] = np.expand_dims(np.dot(gt_img[i], [0.2989, 0.5870, 0.1140]), axis=0)

        sample = {'image': vid_rgb2gray, 'pred': gt_rgb2gray}

        if self.transform is not None:
            sample = self.transform(sample)
        # sample['image']: torch.Size([1, 1, 128, 128, 128])
        # sample['pred']: torch.Size([1])
        return sample['image'], sample['pred']
