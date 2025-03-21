import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.manifold import TSNE

import torch

from algorithms import make_agent
from algorithms.sac import SAC
from unity_config import reach_cfg

colours = ["#8d6a9f", "#8cbcb9", "#dda448", "#bb342f"]

# load obs
def load_obs(filename: str) -> list[np.array]:
    obs_file = filename

    with open(obs_file, 'rb') as f:
        observations = np.load(f)

    return observations

# view one
def view_one_obs(idx: int, observations: list[np.array], fig_num: int = 1) -> matplotlib.figure.Figure:
    fig = plt.figure(fig_num)
    axs = []

    obs = observations[idx]
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(np.moveaxis(obs[i*3:i*3+3], 0, -1))
        axs.append(ax)

    return fig

def load_model(filename: str) -> SAC:
    models_dict = torch.load(filename)

    obs_shape = (9, 84, 84)
    action_shape = (2,)
    action_range = [-1.0, 1.0]

    agent = make_agent(obs_shape, action_shape, action_range, reach_cfg, None)

    agent.actor.load_state_dict(models_dict["actor"])
    agent.critic.load_state_dict(models_dict["critic"])
    agent.critic_target.load_state_dict(models_dict["critic_target"])

    return agent

def infer_one(input_tensor: torch.Tensor, agent: SAC, model_name: str) -> torch.Tensor:
    model = getattr(agent, model_name)
    dist, z = model(input_tensor, None, False, False, True, False)

    return z

def inference(observations: list[np.array], agent: SAC, model_name: str, cameras: list[str]) -> tuple[dict[list[torch.Tensor]]]:
    z_both = {cam: [] for cam in cameras}
    z_shared = {cam: [] for cam in cameras}
    z_private = {cam: [] for cam in cameras}

    print()
    for i, obs in enumerate(observations):
        obs = torch.Tensor(obs).unsqueeze(0).to(device=reach_cfg.device)

        for j, cam in enumerate(cameras):
            z = infer_one(obs[:, j*3:j*3+3], agent, model_name)

            z_both[cam].append(z)
            z_shared[cam].append(z[:, :reach_cfg.feature_dim])
            z_private[cam].append(z[:, reach_cfg.feature_dim:])

            print(f"\rinferred obs {i+1}/{len(observations)},\tcamera {j+1}/{len(cameras)}", end='')
    
    print()

    return z_both, z_shared, z_private

def pca(features: torch.Tensor, n_components: int = 3) -> np.array:
    f = features.detach().cpu().numpy()

    pca = PCA(n_components = n_components)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])

    transformed = pipe.fit_transform(f)

    return transformed

def plot_projections(z, cameras, projection, feature_dim, ax_time = None, ax_cam = None, proj_args = []):
    if not ax_time and not ax_cam:
        fig, (ax_time, ax_cam) = plt.subplots(nrows=2)

    for i, cam in enumerate(cameras):
        features = torch.stack(z[cam]).reshape(-1, feature_dim)
        proj = projection(features, *proj_args)

        times = np.arange(features.shape[0])

        s_time = ax_time.scatter(proj[:, 0], proj[:, 1], c=times)
        s_cam = ax_cam.scatter(proj[:, 0], proj[:, 1], facecolor=colours[i])

    return ax_time, ax_cam

def view(z_both, z_shared, z_private, cameras, projection, feature_dim, proj_args = []):
    fig, axs = plt.subplots(2, 3)

    plot_projections(z_both, cameras, projection, feature_dim*2, axs[0, 0], axs[1, 0], proj_args)
    plot_projections(z_shared, cameras, projection, feature_dim, axs[0, 1], axs[1, 1], proj_args)
    plot_projections(z_private, cameras, projection, feature_dim, axs[0, 2], axs[1, 2], proj_args)

    fig.set_size_inches(14, 9)

if __name__ == '__main__':
    # load observations
    filename = "/home/medcvr/yifei/thesis/data/reach_sim_obs/dvrk_reach_sim_50.npy"
    observations = load_obs(filename)

    # view one
    idx = 15
    fig = view_one_obs(idx, observations, 1)
    # plt.show()

    # load model
    # filename = "/home/medcvr/yifei/thesis/runs/dvrk_reach_mvd/3201550/trained_models/env_step15000/models.pt"
    filename = "/home/medcvr/yifei/MVD/runs/panda_reach_sac_mvd/120801/trained_models/env_step150000/models.pt"
    agent = load_model(filename)

    obs = torch.Tensor(observations[idx][:3]).unsqueeze(0).to(device=reach_cfg.device)
    z = infer_one(obs, agent, "actor")

    # perform inference
    z_both, z_shared, z_private = inference(observations, agent, "actor", ["cam1", "cam2", "cam3"])

    # plot
    fig, axs = plt.subplots(2)
    ax_time, ax_cam = plot_projections(z_both, reach_cfg.cameras, pca, 100, *axs, [2])
    # plt.show()

    view(z_both, z_shared, z_private, reach_cfg.cameras, pca, 50, [2])
    plt.show()

    print(agent)