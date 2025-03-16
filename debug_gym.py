from train import Workspace
from unity_workspace import default_cfg

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import time
import numpy as np
import matplotlib.pyplot as plt

cfg = default_cfg
cfg = default_cfg
cfg.domain_name = "Panda"
cfg.task_name = "PandaReachDense-v3"
cfg.exp_name = "panda_reach_sac_mvd"
cfg.cameras = ["first_person", "third_person_front", "third_person_side"]
cfg.frame_stack = 1
cfg.feature_dim = 50
cfg.eval_on_each_camera = True
cfg.multi_view_disentanglement = True
cfg.image_size = 84

cfg.seed = int(time.time()%10000)

panda_workspace = Workspace(cfg)
panda = panda_workspace.env 

pobs, pinfo = panda.reset()

fig = plt.figure(1)
axs = []

for i in range(3):
    ax = plt.subplot(1, 3, i+1)
    ax.imshow(np.moveaxis(pobs[i*3:i*3+3], 0, -1))
    axs.append(ax)


# panda_workspace.run()

unity_env = UnityEnvironment(f"/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam/reach_mvd_3cam.x86_64", worker_id=2)
unity = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

uobs = np.vstack(unity.reset())

fig = plt.figure(2)
axs = []

for i in range(3):
    ax = plt.subplot(1, 3, i+1)
    ax.imshow(np.moveaxis(uobs[i*3:i*3+3], 0, -1))
    axs.append(ax)

# print(panda)
print(unity)