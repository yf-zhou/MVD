from train import Workspace
from unity_config import default_cfg, reach_cfg

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import time
import numpy as np
import matplotlib.pyplot as plt

def create_panda_workspace():
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

    return panda_workspace, panda

def view_panda_obs(panda):
    pobs, pinfo = panda.reset()

    fig = plt.figure(1)
    axs = []

    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(np.moveaxis(pobs[i*3:i*3+3], 0, -1))
        axs.append(ax)

    plt.show()
    # panda_workspace.run()

def create_unity_workspace(executable=f"/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam_v2/reach_mvd_3cam_v2.x86_64"):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(executable, worker_id=2, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=20, width=168, height=168)
    unity = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

    return unity_env, unity

def view_unity_obs(unity):
    uobs = np.vstack(unity.reset())

    fig = plt.figure(2)
    axs = []

    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(np.moveaxis(uobs[i*3:i*3+3], 0, -1))
        axs.append(ax)

    plt.show()

