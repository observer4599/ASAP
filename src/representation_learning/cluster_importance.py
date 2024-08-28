import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import (get_project_folder, save, layer_init)
import tomli
import gymnasium as gym
from src.environment import make_env
from tqdm import trange
import numpy as np
from src.representation_learning.data import gather_data
import src.ppo as ppo
from collections import defaultdict
from lightning.fabric.loggers import TensorBoardLogger
import statistics
from lightning.fabric import Fabric
from pathlib import Path
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence


class ClusteringImportance(nn.Module):
    def __init__(self, envs, agent, n_clusters: int, n_units: int) -> None:
        super(ClusteringImportance, self).__init__()
        self.input_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_space = envs.single_action_space
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            self.out_dim = self.action_space.n
        else:
            self.out_dim = np.prod(self.action_space.shape)
        self.n_clusters = n_clusters

        # find initial state
        self.base_state = torch.nn.Parameter(
            torch.Tensor(envs.envs[0].get_normalized_obs(
                envs.envs[0].base_state.copy())), requires_grad=False)

        # model modules
        self.agent = agent
        if envs.envs[0].env_id in ("MountainCar-v0",):
            self.encoder = nn.Sequential(layer_init(nn.Linear(self.input_dim, n_units)), nn.Tanh(),
                                         layer_init(nn.Linear(n_units, self.n_clusters), std=1.0))
        else:
            self.encoder = nn.Sequential(layer_init(nn.Linear(self.input_dim, n_units)), nn.Tanh(),
                                         layer_init(
                nn.Linear(n_units, n_units)), nn.Tanh(),
                layer_init(nn.Linear(n_units, self.n_clusters), std=1.0))

        self.importance_func = nn.Sequential(
            layer_init(
                nn.Linear(n_clusters, self.input_dim), std=1.0),
            nn.Sigmoid())

    def forward_agent(self, x, sample: bool = False):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action, output, _, _, _ = self.agent.get_action_and_value(
                x)
            if not sample:
                action = torch.argmax(output, dim=1)
        elif isinstance(self.action_space, gym.spaces.Box):
            action, output, _, _, _ = self.agent.get_action_and_value(
                x)
            if not sample:
                action = output
        else:
            raise NotImplementedError(
                f"Action space: {self.action_space} is not implemented")

        return action, output

    def forward(self, x: torch.Tensor, sample: bool = False):
        logits = self.encoder(x)
        dist = Categorical(logits=logits)

        # new input
        importance = self.importance_func(dist.probs)
        new_in = importance * x + (1 - importance) * self.base_state
        action, output = self.forward_agent(new_in, sample)

        return (action, output, dist, importance)

    def fit_cluster_action(self, obs, output):
        dist = Categorical(logits=self.encoder(obs))
        probs = dist.probs
        cluster_idx = torch.argmax(probs, dim=1)
        cluster_action = []
        for i in range(probs.shape[1]):
            cluster_action.append(output[cluster_idx == i].mean(0))
        self.cluster_action = torch.stack(cluster_action).numpy(force=True)


def train(envs, args, train_b, val_b, fabric,
          model, optimizer):
    training_loader = DataLoader(train_b.get_dataset(),
                                 batch_size=args["batch_size"],
                                 shuffle=True, num_workers=args["n_workers"])
    val_loader = DataLoader(val_b.get_dataset(),
                            batch_size=len(val_b.get_dataset()),
                            shuffle=False)

    training_loader, val_loader = fabric.setup_dataloaders(
        training_loader, val_loader)
    print("Training Dataloader Size:", len(training_loader))

    if isinstance(envs.single_action_space, gym.spaces.Box):
        min_ = torch.FloatTensor(
            envs.single_action_space.low, device=fabric.device)
        max_ = torch.FloatTensor(
            envs.single_action_space.high, device=fabric.device)

    c_coefs = np.geomspace(args["c_coef_start"],
                           args["c_coef_end"], num=args["epochs"])

    for epoch in (pbar := trange(args["epochs"], desc="Epoch")):
        log = defaultdict(list)
        fabric.log("charts/c_coef", c_coefs[epoch], epoch)

        for obs, _, output, _ in training_loader:
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            _, bottleneck_output, dist, importance = model(obs)

            # Policy
            if isinstance(envs.single_action_space, gym.spaces.Discrete):
                pred = Categorical(logits=bottleneck_output).probs
                target = Categorical(logits=output).probs
            elif isinstance(envs.single_action_space, gym.spaces.Box):
                pred = torch.clip(bottleneck_output, min=min_, max=max_)
                target = torch.clip(output, min=min_, max=max_)
            else:
                raise NotImplementedError

            policy_loss = F.mse_loss(pred, target)
            log["loss/policy"].append(policy_loss.item())
            loss = policy_loss

            # Clustering
            # http://proceedings.mlr.press/v48/xieb16.html
            with torch.no_grad():
                weight = dist.probs ** 2 / dist.probs.sum(0, keepdim=True)
                target_dist = Categorical(
                    probs=weight / weight.sum(1, keepdim=True))
            loss_cluster = kl_divergence(p=target_dist, q=dist).mean()

            loss += c_coefs[epoch] * loss_cluster
            log["loss/cluster"].append(loss_cluster.item())

            # Importance
            loss_imp = importance.mean()
            loss += args["imp_coef"] * loss_imp
            log["loss/importance"].append(importance.mean().item())

            fabric.backward(loss)
            optimizer.step()

        log = {key: statistics.mean(dict_v)
               for key, dict_v in log.items()}

        pbar.set_postfix(log)
        pbar.refresh()
        fabric.log_dict(log, step=epoch)

        # Validation
        if epoch % args["val_interval"] == 0:
            val_log = defaultdict(list)
            with torch.no_grad():
                for val_obs, _, val_output, _ in val_loader:
                    _, bottleneck_output, _, _ = model(val_obs)

                    if isinstance(envs.single_action_space, gym.spaces.Discrete):
                        pred = Categorical(logits=bottleneck_output).probs
                        target = Categorical(logits=val_output).probs
                    elif isinstance(envs.single_action_space, gym.spaces.Box):
                        pred = bottleneck_output
                        target = val_output
                    else:
                        raise NotImplementedError
                    val_log["loss/val_policy"].append(
                        F.mse_loss(pred, target).item())
            val_log = {key: statistics.mean(dict_v)
                       for key, dict_v in val_log.items()}
            fabric.log_dict(val_log, step=epoch)

    return model


def main(args):
    # torch.use_deterministic_algorithms(args["torch_deterministic"])
    if "model_load_path" in args:
        fabric = Fabric(accelerator=args["accelerator"])
    else:
        log_folder = get_project_folder() / f"runs/{args['task']}"
        logger = TensorBoardLogger(root_dir=log_folder,
                                   name=f"{args['algorithm']}__{args['env_id']}")
        logger.experiment.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in args.items()])),
        )

        fabric = Fabric(accelerator=args["accelerator"],
                        loggers=logger)
    fabric.seed_everything(args["seed"], workers=True)

    checkpoint_path = get_project_folder(
    ) / f'runs/train/{args["algorithm"]}__{args["env_id"]}/{args["policy_load_path"]}'
    buffer_path = checkpoint_path.parent / "rollout_buffer.pickle"

    if "model_load_path" in args:
        checkpoint_path = get_project_folder(
        ) / f'runs/clustering/{args["algorithm"]}__{args["env_id"]}/{args["model_load_path"]}'

    checkpoint = torch.load(checkpoint_path)

    if checkpoint["envs"] is not None and "gamma" in checkpoint["envs"]:
        gamma = checkpoint["envs"]["gamma"][0]
    else:
        gamma = 0.99
    envs = gym.vector.SyncVectorEnv(
        [make_env(args["env_id"], args["seed"] + i, i, False, gamma,
                  normalize=False)
         for i in range(args["num_envs"])]
    )

    if checkpoint["envs"] is not None:
        for i, env in enumerate(envs.envs):
            j = i % len(checkpoint["envs"]["obs_mean"])
            env.obs_rms.mean = checkpoint["envs"]["obs_mean"][j]
            env.obs_rms.var = checkpoint["envs"]["obs_var"][j]
            env.obs_rms.count = checkpoint["envs"]["obs_count"][j]
            if "return_var" in checkpoint["envs"]:
                env.return_rms.mean = checkpoint["envs"]["return_mean"][j]
                env.return_rms.var = checkpoint["envs"]["return_var"][j]
                env.return_rms.count = checkpoint["envs"]["return_count"][j]

    # Agent setup
    agent = ppo.Agent(envs)
    if "model_load_path" not in args:
        agent.load_state_dict(checkpoint["model"])

    # Data gathering
    train_b, val_b, test_b = gather_data(
        args, buffer_path, envs, agent, fabric.device)

    model = ClusteringImportance(envs=envs,
                                 n_clusters=args["n_clusters"],
                                 agent=agent, n_units=args["n_units"])
    if "model_load_path" in args:
        model.load_state_dict(checkpoint["model"])
        return envs, model, fabric, (train_b, val_b, test_b)

    # Set up optimizer
    params = []
    for name, param in model.named_parameters():
        if "agent" not in name:
            params.append(param)
    optimizer = torch.optim.Adam(params, lr=args["learning_rate"])
    model, optimizer = fabric.setup(model, optimizer)

    model = train(envs, args, train_b,
                  val_b,
                  fabric, model, optimizer)

    save(Path(logger.experiment.log_dir), envs, model)
    return envs, model, fabric, (train_b, val_b, test_b)


def get_args(env_id: str = "MountainCar-v0"):
    if len(env_id) != 0:
        env_id = "_" + env_id
    config_file = get_project_folder(
    ) / f"configs/train_representation{env_id}.toml"
    with config_file.open("rb") as f:
        args = tomli.load(f)
    return args


if __name__ == "__main__":
    # Config file loading
    torch.set_float32_matmul_precision("high")
    main(get_args())
