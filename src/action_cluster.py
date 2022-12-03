import matplotlib.pyplot as plt
from runners.episode_runner import EpisodeRunner
from modules.action_encoders.obs_reward_encoder import ObsRewardEncoder
from main import _get_config, recursive_dict_update
import torch as th
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from copy import deepcopy
import sys
import os
import yaml
from types import SimpleNamespace as SN
import numpy as np

class ActionCluster:
    def __init__(self, args, n_agents=20):
        runner = EpisodeRunner(args=args, logger=None)

        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]

        self.n_clusters = args.n_role_clusters
        self.n_actions = args.n_actions
        self.discrete_steering_dim = args.env_args['discrete_steering_dim']
        obsRewardEncoder = ObsRewardEncoder(args)
        obsRewardEncoder.load_state_dict(th.load(args.action_encoder_weight_path))
        self.action_encoder = obsRewardEncoder

    def cluster_actions(self):
        action_repr = self.action_encoder()
        self.action_repr_array = action_repr.detach().cpu().numpy()  # [n_actions, action_latent_d]

        self.k_means = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.action_repr_array)
        print((self.k_means.labels_))

    def plot(self):
        action_tsne = TSNE(n_components=3, random_state=0).fit_transform(self.action_repr_array)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(self.n_clusters):
            action_cluster = []
            for action in range(self.n_actions):
                if self.k_means.labels_[action] == i:
                    action_cluster.append(action_tsne[action])

                    steering = action % self.discrete_steering_dim
                    throttle = action // self.discrete_steering_dim
                    ax.text(action_tsne[action][0], action_tsne[action][1], action_tsne[action][2], (steering, throttle))

            action_cluster = np.array((action_cluster))

            ax.scatter(action_cluster[:, 0], action_cluster[:, 1], action_cluster[:, 2])


        plt.show()


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    cluster = ActionCluster(SN(**config_dict))
    cluster.cluster_actions()
    cluster.plot()
    # steering = float(action % self.discrete_steering_dim) * self.steering_unit - 1.0
    # 40
    # throttle = float(action // self.discrete_steering_dim) * self.throttle_unit - 1.0