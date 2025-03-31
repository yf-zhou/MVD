import torch
from torch.utils.data import Dataset 
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image 

import os
import logging
import numpy as np

from unity_config import Conf

class SimRealDataset(Dataset):
    def __init__(self, cfg: Conf, transform=None, logger=logging.getLogger(__name__)):
        self.cfg = cfg
        self.logger = logger
        
        self.root = cfg.data_folder
        self.directories = [f.name for f in cfg.folders]
        self.cameras = cfg.cameras
        
        self.transform = transform
        
        self.image_dims = tuple(cfg.image_dims)
        
        self.time_step = cfg.time_step
        self.tail_length = self.time_step #+ 1   # due to numbering
        
        self.lengths = {d: len(os.listdir(os.path.join(self.root, d, self.cameras[0]))) - self.tail_length for d in self.directories}
        self.starts = {self.directories[0]: 0}
        for i, d in enumerate(self.directories[1:]):
            self.starts[d] = self.starts[self.directories[i]] + self.lengths[self.directories[i]]
        
    def __len__(self):
        return sum(self.lengths.values())
    
    def __getitem__(self, idx):
        directory = self.directory_from_idx(idx)
        start = self.starts[directory]
        
        idcs = [idx - start + 1, idx - start + self.time_step + 1]  # numbering in folder starts at 1
        
        images = []

        for cam in self.cameras:
            img_path = os.path.join(self.root, directory, cam, f"img_{idcs[0]:04}.png")
            image = decode_image(img_path)

            if self.transform:
                image = self.transform(image)

            images.append(image)

        obs = torch.stack(images).reshape(-1, *self.image_dims[1:])

        images = []

        for cam in self.cameras:
            img_path = os.path.join(self.root, directory, cam, f"img_{idcs[1]:04}.png")
            image = decode_image(img_path)

            if self.transform:
                image = self.transform(image)

            images.append(image)

        next_obs = torch.stack(images).reshape(-1, *self.image_dims[1:])

        return obs, next_obs, idcs
    
    def directory_from_idx(self, idx):
        for i, d in enumerate(self.directories[:-1]):
            if idx in range(self.starts[d], self.starts[self.directories[i+1]]):
                return d 
        return self.directories[-1]
