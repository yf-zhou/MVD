import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter

import os 
import yaml 
import logging 
import argparse
import numpy as np
import matplotlib.pyplot as plt

from unity_config import reach_cfg

from algorithms.models import Actor
from algorithms.info_nce import InfoNCE
from unity_config import Conf 
from simrealdataset import SimRealDataset

# logging
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

file_handler = logging.FileHandler("train.log", mode='a', encoding='utf-8')
formatter = logging.Formatter(
    "[{asctime} - {levelname}]: {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
file_handler.setLevel("DEBUG")
logger.addHandler(file_handler)

writer = SummaryWriter()

def update(model: Actor, optimizer, obs, next_obs, idcs, epoch, step):
    z, z_shared, z_private = model(obs)
    next_z, next_z_shared, next_z_private = model(next_obs)

    num_cameras = z.shape[1]

    shared_loss = InfoNCE(negative_mode="mixed")
    private_loss = InfoNCE(negative_mode="paired")

    pos_idx = np.random.randint(num_cameras)
    anchor_idxs = [i for i in range(num_cameras) if i != pos_idx]
    multi_view_loss = 0

    for anchor_idx in anchor_idxs:
        shared_query = z_shared[:, anchor_idx]
        shared_positive_key = z_shared[:, pos_idx]
        shared_negative_key = z_private 
        multi_view_loss += shared_loss(shared_query, shared_positive_key, shared_negative_key)

        private_query = z_private[:, anchor_idx]
        private_positive_key = next_z_private[:, pos_idx]
        private_negative_key = z_private[:, [pos_idx]]
        multi_view_loss += private_loss(private_query, private_positive_key, private_negative_key)

    multi_view_loss /= len(anchor_idxs)

    optimizer.zero_grad()
    multi_view_loss.backward()
    optimizer.step()

    logger.info(f"{epoch=}\t{step=}\t{idcs=}\t{multi_view_loss=}")
    writer.add_scalar('loss/train', multi_view_loss, step)
    print(f"epoch: {epoch}\tstep: {step:04}\tloss: {multi_view_loss:.4f}")
    return multi_view_loss

def train(model: Actor, dataset: SimRealDataset, cfg: Conf):
    batch_size = cfg.batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=cfg.shuffle, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    iters = []
    losses = []

    i = 0
    for epoch in range(cfg.num_epochs):
        j = 0
        for imgs, idcs in iter(dataloader):
            if cfg.device == 'cuda' and torch.cuda.is_available():
                imgs = imgs.cuda()
                idcs = idcs.cuda()
            
            loss = update(model, optimizer, imgs[:, :, 0], imgs[:, :, 1], idcs, epoch, j)

            iters.append(i)
            losses.append(float(loss)/batch_size)

            i += 1
            j += 1

    torch.save(model.state_dict(), os.path.join(cfg.model_folder, "actor.pt"))

    writer.flush()

    plt.figure()
    plt.plot(iters, losses)
    plt.savefig(os.path.join(cfg.output_folder, "losses.png"))

def run_train(cfg: Conf):
    logger.info(f"training config: {cfg=}")

    model = Actor(obs_shape=cfg.image_dims,
                  action_shape=cfg.action_shape,
                  cfg=cfg)
    
    if cfg.device == 'cuda' and torch.cuda.is_available():
        model.cuda()
        print("Training on GPU")
    else:
        print("Training on CPU")

    transform = v2.Compose([])
    dataset = SimRealDataset(transform=transform, cfg=cfg, logger=logger)

    train(model, dataset, cfg)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        help="yaml file with configuration for training"
    )
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
        
    run_train(cfg)
