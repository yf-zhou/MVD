import numpy as np

from debug_gym import create_unity_workspace, create_panda_workspace

def collect_unity_obs(total_obs, executable_name, save_dir, save_file):
    _, unity = create_unity_workspace(executable_name)

    all_obs = []

    obs = np.vstack(unity.reset())
    all_obs.append(obs)

    action = [-0.5, 0.25]

    for i in range(total_obs):
        # obs = np.vstack(unity.reset())
        obs, _, _, _ = unity.step(action)

        all_obs.append(np.vstack(obs))

    unity.close()

    with open(save_dir+save_file, 'wb') as f:
        np.save(f, all_obs)

def collect_panda_obs(total_obs, save_dir, save_file):
    _, panda = create_panda_workspace()

    all_obs = []

    obs, _ = panda.reset()
    all_obs.append(obs)

    action = [0.1, 0.1, -0.1]
    
    for i in range(total_obs):
        # obs, _ = panda.reset()
        obs, _, _, _, _ = panda.step(action)

        all_obs.append(obs)

    with open(save_dir+save_file, 'wb') as f:
        np.save(f, all_obs)

if __name__ == '__main__':
    total_obs = 50

    executable = "/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam_v2/reach_mvd_3cam_v2.x86_64"

    save_dir = "/home/medcvr/yifei/thesis/data/reach_sim_obs/"
    save_file = "dvrk_reach_sim_50_steps.npy"

    collect_unity_obs(total_obs, executable, save_dir, save_file)

    # total_obs = 50

    # save_dir = "/home/medcvr/yifei/thesis/data/panda_reach_obs/"
    # save_file = "panda_reach_50_steps.npy"

    # collect_panda_obs(total_obs, save_dir, save_file)