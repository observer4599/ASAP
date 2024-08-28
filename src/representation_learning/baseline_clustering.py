from sklearn.metrics import silhouette_score
import cluster_importance as ci
import warnings
from torch.distributions.categorical import Categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statistics
from collections import defaultdict
import copy
import numpy as np
from sklearn.cluster import KMeans
from captum.attr import ShapleyValueSampling, FeatureAblation, DeepLift
from src.representation_learning.data import gather_data
from src import ppo
import torch
from src.environment import make_env
import gymnasium as gym
from src.utils import get_project_folder
from lightning.fabric import Fabric
import argparse


class MyFeatureAttribution:
    def __init__(self, forward_func) -> None:
        self.forward_func = forward_func

    def attribute(self, x, baseline, raw=True):
        importances = []
        output = self.forward_func(x, raw=raw)
        for f in range(x.shape[1]):
            x_mod = x.clone()
            x_mod[:, f] = baseline[:, f]
            output_mod = self.forward_func(x_mod, raw=raw)
            importance = torch.square(output - output_mod).sum(1)
            importances.append(importance)

        return torch.stack(importances, dim=1)


class CInstance:
    def __init__(self, agent, baseline, type_: str, attr_type: str,
                 seed: int) -> None:
        self.type_ = type_
        self.attr_type = attr_type
        self.agent = agent
        self.baseline = baseline
        self.seed = seed

        self.value_scaler = StandardScaler()
        self.attr_scaler = MinMaxScaler()
        self.attr_ss_scaler = StandardScaler()

        class AttrForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor):
                output = agent(x, raw=True)
                if agent.discrete:
                    return torch.max(output, dim=1).values
                return output.mean(1)

        attr_model = AttrForward()

        match attr_type:
            case "ablation":
                self.ablator = FeatureAblation(attr_model)
            case "shapley":
                self.ablator = ShapleyValueSampling(attr_model)
            case "backprop":
                self.ablator = DeepLift(attr_model)
            case "my_ablation":
                self.ablator = MyFeatureAttribution(agent)
            case "none":
                self.ablator = None
            case _:
                raise NotImplementedError(f"{attr_type} is not valid")

    def compute_attr(self, x: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                attr = self.ablator.attribute(x.clone(), self.baseline)
        return attr.detach()

    def compute_input(self, obs: torch.Tensor, output: torch.Tensor,
                      value: torch.Tensor, activation: torch.Tensor) -> np.ndarray:

        # standardlize value
        value = value.detach().cpu().numpy()
        if hasattr(self.value_scaler, "feature_names_in_"):
            value = self.value_scaler.transform(value)
        else:
            value = self.value_scaler.fit_transform(value)
        value = torch.from_numpy(value)

        if self.ablator is not None:
            attr = self.compute_attr(obs)
            if not hasattr(self.attr_scaler, "n_features_in_"):
                self.attr_scaler.fit(attr.numpy(force=True))
                self.attr_ss_scaler.fit(attr.numpy(force=True))

        match self.type_:
            case "activation":
                input_ = activation
            case "value":
                input_ = value
            case "obs":
                input_ = obs
            case "action":
                input_ = output
            case "attr":
                input_ = attr
            case _:
                raise NotImplementedError(
                    f"{self.type_} is not a valid input type.")

        return input_.detach().cpu().numpy()

    def fit(self, obs: torch.Tensor, output: torch.Tensor,
            value: torch.Tensor, activation: torch.Tensor,
            n_clusters: int) -> None:
        self.n_clusters = n_clusters
        x = self.compute_input(obs, output, value, activation)
        self.c_instance = KMeans(
            n_clusters=n_clusters, random_state=self.seed, n_init="auto")
        cluster_idx = self.c_instance.fit_predict(x)

        cluster_action, cluster_value = [], []
        for i in range(n_clusters):
            # action
            cluster_output = output[cluster_idx == i].cpu().numpy()
            cluster_action.append(np.mean(cluster_output, axis=0))
            # value
            cluster_value.append(
                np.mean(value[cluster_idx == i].cpu().numpy()))
        self.cluster_action = np.stack(cluster_action, axis=0)
        self.cluster_value = np.array(cluster_value)

    def predict(self, x: torch.Tensor, output: torch.Tensor,
                value: torch.Tensor, activation: torch.Tensor):
        x = self.compute_input(x, output, value, activation)
        cluster_idx = self.c_instance.predict(x)
        return (cluster_idx,
                np.take(self.cluster_action, cluster_idx, axis=0),
                np.take(self.cluster_value, cluster_idx))


def get_activation(obs, agent):
    activation = obs
    for i in range(4):
        if agent.discrete:
            activation = agent.actor[i](activation)
        else:
            activation = agent.actor_mean[i](activation)

    return activation


def test_cluster_policy_online(envs, fabric, cluster_instance, seed: int,
                               n_ep: int = 10):
    envs = copy.deepcopy(envs)
    fabric.seed_everything(seed, workers=True)

    ep_info = defaultdict(list)
    curr_ep = 0

    obs = torch.FloatTensor(envs.reset(seed=seed)[0],
                            device=fabric.device)
    while curr_ep < n_ep:
        if isinstance(cluster_instance, ci.ClusteringImportance):
            cluster_idx = np.argmax(cluster_instance(
                obs)[2].probs.numpy(force=True))
            action = cluster_instance.cluster_action[cluster_idx][np.newaxis, ...]
        else:
            _, output, _, _, value = cluster_instance.agent.get_action_and_value(
                obs)
            activation = get_activation(obs, cluster_instance.agent)
            _, action, _ = cluster_instance.predict(
                obs, output, value, activation)
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            action = np.argmax(action, axis=1)
        next_obs, _, _, _, infos = envs.step(action)
        obs = torch.FloatTensor(next_obs, device=fabric.device)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue
        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            ep_info["ep_length"].append(info["episode"]["l"].item())
            ep_info["ep_return"].append(info["episode"]["r"].item())
            curr_ep += 1
            obs = torch.FloatTensor(envs.reset(seed=curr_ep + seed)[0],
                                    device=fabric.device)
    final_info = {"mean_return": statistics.mean(ep_info["ep_return"]),
                  "std_return": statistics.stdev(ep_info["ep_return"])}

    return final_info


def test_masked(envs, fabric, cluster_instance, seed: int, n_ep: int = 10,
                attr_on: bool = True):
    envs = copy.deepcopy(envs)
    fabric.seed_everything(seed, workers=True)

    ep_info = defaultdict(list)
    curr_ep = 0

    obs = torch.FloatTensor(envs.reset(seed=seed)[0],
                            device=fabric.device)
    while curr_ep < n_ep:
        if hasattr(cluster_instance, "ablator") and cluster_instance.ablator is not None:
            attr = cluster_instance.compute_attr(obs)
            importance = cluster_instance.attr_scaler.transform(attr)
            importance = torch.clamp(
                torch.FloatTensor(importance), min=0.0, max=1.0)
            obs = obs * importance + (1 - importance) * \
                cluster_instance.baseline
            ep_info["importance"].append(importance.mean().item())

        if isinstance(cluster_instance, ci.ClusteringImportance):
            output = cluster_instance(obs)[1]
        else:
            output = cluster_instance.agent.get_action_and_value(obs)[1]
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            action = torch.argmax(
                output, dim=1)
        else:
            action = output
        action = action.numpy(force=True)
        next_obs, _, _, _, infos = envs.step(action)
        obs = torch.FloatTensor(next_obs, device=fabric.device)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue
        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            ep_info["ep_return"].append(info["episode"]["r"].item())
            ep_info["ep_length"].append(info["episode"]["l"].item())
            curr_ep += 1
            obs = torch.FloatTensor(envs.reset(seed=curr_ep+seed)[0],
                                    device=fabric.device)
    final_info = {"mean_return": statistics.mean(ep_info["ep_return"]),
                  "stdev_return": statistics.stdev(ep_info["ep_return"]),
                  }

    return final_info


def test_sihouette(obs, output, value, activation, cluster_instance):
    if isinstance(cluster_instance, ci.ClusteringImportance):
        probs = cluster_instance(obs)[2].probs
        cluster_idx = torch.argmax(probs, dim=1).numpy(force=True)
    else:
        cluster_idx, _, _ = cluster_instance.predict(
            obs, output, value, activation)

    return silhouette_score(obs.numpy(force=True), cluster_idx, random_state=0)


def get_data(envs, fabric, buffer):
    obs = buffer.get_dataset().obs.to(fabric.device)
    output = buffer.get_dataset().output.to(fabric.device)
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        output = Categorical(logits=output).probs
    value = buffer.get_dataset().value.to(fabric.device).unsqueeze(1)
    return obs, output, value


def main(args):
    fabric = Fabric(accelerator=args.accelerator)
    fabric.seed_everything(args.seed, workers=True)

    checkpoint_path = get_project_folder(
    ) / f'runs/train/{args.algorithm}__{args.env_id}/{args.agent_path}'
    buffer_path = checkpoint_path.parent / "rollout_buffer.pickle"
    checkpoint = torch.load(checkpoint_path)

    # Environment Setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, 0.99)
         for i in range(args.num_envs)]
    )

    if checkpoint["envs"] is not None:
        for i, env in enumerate(envs.envs):
            j = i % len(checkpoint["envs"]["obs_mean"])
            env.obs_rms.mean = checkpoint["envs"]["obs_mean"][j]
            env.obs_rms.var = checkpoint["envs"]["obs_var"][j]
            env.obs_rms.count = checkpoint["envs"]["obs_count"][j]
            env.normalize_obs = False
            if hasattr(env, "reward_var"):
                env.reward_var = checkpoint["envs"]["reward_var"][j]
                env.gamma = checkpoint["envs"]["gamma"][j]
                env.normalize_rews = False

    # Agent setup
    agent = ppo.Agent(envs)
    agent.load_state_dict(checkpoint["model"])

    # Data gathering
    train_b, val_b, test_b = gather_data(
        args, buffer_path, envs, agent, fabric.device)

    train_obs, train_output, train_value = get_data(envs, fabric, train_b)
    train_activation = get_activation(train_obs, agent)

    val_obs, val_output, val_value = get_data(envs, fabric, val_b)
    val_activation = get_activation(val_obs, agent)

    test_obs, test_output, test_value = get_data(envs, fabric, test_b)
    test_activation = get_activation(test_obs, agent)

    # Get our method
    model = ci.main(ci.get_args(args.env_id))[1]
    model.fit_cluster_action(train_obs, train_output)

    # Clustering
    baseline = envs.envs[0].get_normalized_obs(envs.envs[0].base_state)
    baseline = torch.FloatTensor(baseline, device=fabric.device)

    attr_types = ("none",) if args.type != "attr" else (
        "ablation", "shapley", "backprop",)
    cluster_instances = dict()
    cluster_instances["model"] = model
    for attr_type in attr_types:
        cluster_instances[attr_type] = CInstance(agent, baseline,
                                                 args.type, attr_type, args.seed)
        cluster_instances[attr_type].fit(
            train_obs, train_output, train_value, train_activation,
            args.num_clusters)

    print(f"# --- {args.env_id} ---")
    for attr_type in cluster_instances.keys():
        print(f"# --- {attr_type} ---")
        silhouette = test_sihouette(test_obs, test_output,
                                    test_value, test_activation, cluster_instances[attr_type])
        print(f"# Silhouette: {silhouette}")
        det_action = test_cluster_policy_online(
            envs, fabric, cluster_instances[attr_type], args.seed, 100)
        print(
            f"# Deterministic Action Return - mean: {det_action['mean_return']}, std: {det_action['std_return']}")
        masked_return = test_masked(envs, fabric,
                                    cluster_instances[attr_type], args.seed, 100)
        print(
            f"# Masked Return - mean: {masked_return['mean_return']}, std: {masked_return['stdev_return']}")

    return args, fabric, cluster_instances, envs, (train_b, val_b, test_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--accelerator", type=str, default="cpu",
                        help="TODO")

    parser.add_argument("--algorithm", type=str, default="ppo",
                        help="TODO")
    parser.add_argument("--agent-path", type=str, default="version_0/model.pt",
                        help="TODO")
    parser.add_argument("--env-id", type=str,
                        help="the id of the environment", default="MountainCar-v0",
                        choices=["Acrobot-v1", "CartPole-v1", "FlappyBird-v0", "MountainCar-v0", "Swimmer-v4"])
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")

    parser.add_argument("--num-clusters", type=int, default=11,
                        help="the number of clusters to create")
    parser.add_argument("--type", type=str, default="value",
                        choices=["obs", "value", "activation", "attr", "action"])
    parser.add_argument("--train-bs", type=int, default=200_000)
    parser.add_argument("--train-collect-samples", type=int, default=100_000)
    parser.add_argument("--val-bs", type=int, default=40_000)
    parser.add_argument("--val-collect-samples", type=int, default=20_000)
    parser.add_argument("--test-bs", type=int, default=40_000)
    parser.add_argument("--test-collect-samples", type=int, default=20_000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    ARGS = parse_args()
