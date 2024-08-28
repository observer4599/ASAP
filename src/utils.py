from pathlib import Path
import random
import numpy as np
import torch
import argparse
from copy import deepcopy
from collections import defaultdict

Tensor = torch.Tensor
Array = np.ndarray
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_project_folder() -> Path:
    return Path(__file__).parent.parent


def seed_everything(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                        help="the name of the config file to use")
    return parser.parse_args()


def save(folder, envs, model) -> None:
    if hasattr(envs.envs[0], "obs_rms"):
        envs_param = defaultdict(list)
        for env in envs.envs:
            envs_param["obs_mean"].append(env.obs_rms.mean.copy())
            envs_param["obs_var"].append(env.obs_rms.var.copy())
            envs_param["obs_count"].append(deepcopy(env.obs_rms.count))
            if hasattr(env, "return_rms"):
                envs_param["return_mean"].append(
                    env.return_rms.mean.copy())
                envs_param["return_var"].append(
                    env.return_rms.var.copy())
                envs_param["return_count"].append(
                    deepcopy(env.return_rms.count))
                envs_param["gamma"].append(env.gamma)
    else:
        envs_param = None

    save_path = folder / f"model.pt"
    torch.save({"model": model.state_dict(),
                "envs": envs_param}, save_path)
