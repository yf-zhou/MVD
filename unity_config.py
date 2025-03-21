import time
from datetime import datetime

class Conf:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Conf(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Conf(v) if isinstance(v, dict) else v)

default_cfg = Conf({
    "algorithm": "sac",
    "unity_build": f"/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam/reach_mvd_3cam.x86_64",
    "log_dir": "runs",
    "exp_name": "dvrk_reach",
    "seed": int(time.time()%10000), 
    "device": "cuda",
    "build_executable": "/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam/reach_mvd_3cam.x86_64",
    "cameras": ["cam1", "cam2", "cam3"],
    "replay_buffer_capacity": 100000,
    "image_pad": 4,
    "num_eval_episodes": 10,
    "num_train_steps": 1000, 
    "eval_freq": 1000,
    "save_video": False,
    "eval_on_each_camera": True,
    "num_seed_steps": 1000,
    "discount": 0.99,
    "critic_tau": 0.01,
    "encoder_tau": 0.05,
    "actor_update_freq": 2,
    "critic_target_update_freq": 2,
    "batch_size": 128,
    "image_reconstruction_loss": True,
    "decoder_update_freq": 1,
    "mvd_update_freq": 2,
    "multi_view_disentanglement": True,
    "mvd_lr": 1e-3,
    "mvd_beta": 0.9,
    "init_temperature": 0.1,
    "actor_lr": 1e-3,
    "alpha_lr": 1e-4,
    "critic_lr": 1e-3,
    "decoder_weight_lambda": 1e-7,
    "actor_log_std_min": -10,
    "actor_log_std_max": 2,
    "frame_stack": 3,
    "use_proprioceptive_state": False,
    "feature_dim": 50,
    "hidden_dim": 1024,
    "hidden_depth": 2,
    "num_conv_layers": 4,
    "num_filters": 32,
    "log_freq": 1000, 
    "action_repeat": 4,
    "domain_name": "Unity",
    "save_freq": 250000
})

reach_cfg = Conf({
    "domain_name": "Unity", 
    "task_name": "dvrk_reach",
    "exp_name": "dvrk_reach_mvd",
    "build_executable": f"/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam_v2/reach_mvd_3cam_v2.x86_64",
    "device": "cuda",
    "seed": int(datetime.now().strftime("%m%d%H%M")), 

    "cameras": ["cam1", "cam2", "cam3"],
    "multi_view_disentanglement": True,
    "eval_on_each_camera": False,

    "algorithm": "sac",
    "action_repeat": 1,
    "num_train_steps": 250000,
    # "num_train_steps": 100,
    "num_train_iters": 1,
    "replay_buffer_capacity": 100000,
    "num_seed_steps": 1000,

    "image_size": 64,
    "frame_stack": 1,
    "image_pad": 4,
    "use_proprioceptive_state": False,

    "eval_freq": 10000,
    # "eval_freq": 50,
    "num_eval_episodes": 20,

    "log_freq": 1000,
    "save_freq": 125000,
    # "save_freq": 50,
    "log_dir": "runs",
    "save_video": True,

    "discount": 0.99,
    "batch_size": 128,
    "hidden_dim": 1024,
    "hidden_depth": 2,

    "actor_lr": 1e-3,
    "actor_beta": 0.9,
    "actor_log_std_min": -10,
    "actor_log_std_max": 2,
    "actor_update_freq": 2,
    "init_temperature": 0.1,
    "alpha_lr": 1e-4,

    "critic_lr": 1e-3,
    "critic_tau": 0.01,
    "critic_target_update_freq": 2,

    "encoder_tau": 0.05,
    "num_conv_layers": 4,
    "feature_dim": 50,
    "num_filters": 32,
    "image_reconstruction_loss": True,
    "decoder_weight_lambda": 1e-7,
    "decoder_update_freq": 1,

    "mvd_lr": 1e-3,
    "mvd_beta": 0.9,
    "mvd_update_freq": 2
})