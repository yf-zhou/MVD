import time

if __name__ == '__main__':
    from unity_workspace import default_cfg
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

    from train import Workspace as W
    global workspace
    workspace = W(cfg)
    start_time = time.time()
    workspace.run()
    print("total run time: ", time.time()-start_time)