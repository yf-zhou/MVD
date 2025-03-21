import numpy as np

from debug_gym import create_unity_workspace

def collect_obs(total_obs, executable_name, save_dir, save_file):
    _, unity = create_unity_workspace(executable_name)

    all_obs = []

    for i in range(total_obs):
        obs = np.vstack(unity.reset())

        all_obs.append(obs)

    unity.close()

    with open(save_dir+save_file, 'wb') as f:
        np.save(f, all_obs)

if __name__ == '__main__':
    total_obs = 50

    executable = "/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam_v2/reach_mvd_3cam_v2.x86_64"

    save_dir = "/home/medcvr/yifei/thesis/data/reach_sim_obs/"
    save_file = "dvrk_reach_sim_50.npy"

    collect_obs(total_obs, executable, save_dir, save_file)