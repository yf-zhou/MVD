from train import Workspace
from unity_workspace import default_cfg

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import time

cfg = default_cfg
cfg.domain_name = "Panda"
cfg.task_name = "PandaReachDense-v3"
cfg.exp_name = "panda_reach_sac_mvd"
cfg.image_size = 84
cfg.cameras = ["first_person", "third_person_front", "third_person_side"]

cfg.seed = int(time.time()%10000)

panda_workspace = Workspace(cfg)
panda = panda_workspace.env 

panda_workspace.run()

unity_env = UnityEnvironment(f"/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam/reach_mvd_3cam.x86_64")
unity = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)


print(panda)