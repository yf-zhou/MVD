import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import decode_image
import torchvision.transforms.v2 as v2

from unity_config import Conf
from simrealdataset import SimRealDataset

if __name__ == '__main__':
    # directory = "data/10"
    # cameras = ["sim_fixed", "oak"]

    span = range(30, 130, 2)

    save_dir = "data/reach_real_obs/"
    save_file = "dvrk_reach_real_10.npy"

    # files = sorted(os.listdir(os.path.join(directory), cameras[0]))

    # all_obs = []

    # for i in span:
    #     image = decode_image()

    config_file = "config/reach.yaml"

    with open(config_file, 'r') as f:
        cfg = Conf(yaml.safe_load(f))

    transform = v2.Compose([
        v2.RandomResizedCrop(size=cfg.image_dims[1:], scale=(1, 1), ratio=[1, 1])
    ])
    dataset = SimRealDataset(transform=transform, cfg=cfg)

    all_obs = []
    for i in span:
        obs, next_obs, idcs = dataset[i]

        all_obs.append(obs)

    with open(save_dir+save_file, 'wb') as f:
        np.save(f, all_obs)