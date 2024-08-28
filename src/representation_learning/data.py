# RolloutBuffer adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

import gymnasium as gym
import torch
from torch.utils.data import Dataset
import pickle
from tqdm import trange
from copy import deepcopy
from lightning import fabric


class RolloutDataset(Dataset):
    def __init__(self, envs, obs, action, output, value) -> None:
        action_space = envs.single_action_space
        self.obs = obs.clone().reshape((-1,) + envs.single_observation_space.shape)
        self.action = action.clone().reshape((-1,) + action_space.shape)
        self.value = value.clone().reshape(-1)

        if isinstance(action_space, gym.spaces.Discrete):
            self.output = output.clone().reshape(
                (-1,) + (action_space.n,))
        elif isinstance(action_space, gym.spaces.Box):
            self.output = output.clone().reshape(
                (-1,) + action_space.shape)
        else:
            raise NotImplementedError(
                f"Action space: {action_space} is not implemented")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx: int):
        return (self.obs[idx], self.action[idx], self.output[idx],
                self.value[idx])


class RolloutBuffer:
    def __init__(self, envs: gym.vector.SyncVectorEnv, buffer_size: int,
                 seed: int) -> None:
        self.buffer_size = buffer_size
        self.seed = seed

        self.envs = deepcopy(envs)
        action_space = envs.single_action_space
        observation_space = envs.single_observation_space

        self.n_envs = len(envs.envs)
        dim = (int(self.buffer_size / self.n_envs), self.n_envs)
        self.obs = torch.zeros(
            dim + observation_space.shape, dtype=torch.float32)
        self.action = torch.zeros(
            dim + action_space.shape, dtype=torch.float32)
        self.value = torch.zeros(dim, dtype=torch.float32)

        if isinstance(action_space, gym.spaces.Discrete):
            output_dim = dim + (action_space.n,)
        elif isinstance(action_space, gym.spaces.Box):
            output_dim = dim + action_space.shape
        else:
            raise NotImplementedError(
                f"Action space: {action_space} is not implemented")

        self.output = torch.zeros(output_dim, dtype=torch.float32)

        self.pos = 0
        self.full = False
        self.updated = False

    def add(self, obs: torch.Tensor, action: torch.Tensor,
            output: torch.Tensor, value: torch.Tensor) -> None:
        self.obs[self.pos] = obs.clone()
        self.action[self.pos] = action.clone()
        self.output[self.pos] = output.clone()
        self.value[self.pos] = value.clone()

        self.pos += 1
        if self.buffer_size == self.pos:
            self.full = True
            self.pos = 0

    def collect_data(self, agent, model, nsamples: int, device):
        fabric.seed_everything(seed=self.seed)
        envs = deepcopy(self.envs)
        agent = deepcopy(agent)
        model = model if model is None else deepcopy(model)
        self.updated = True
        obs = torch.FloatTensor(envs.reset(seed=self.seed)[0],
                                device=device)
        for _ in (pbar := trange(nsamples, leave=False)):
            with torch.no_grad():
                action, output, _, _, value = agent.get_action_and_value(
                    obs)

            action = action.cpu()
            self.add(obs=obs.cpu(),
                     action=action,
                     output=output.cpu(),
                     value=value.squeeze(1).cpu())
            action = action.numpy()

            if model is not None:
                with torch.no_grad():
                    action = model(obs, True)[0].cpu().numpy()

            next_obs, _, _, _, _ = envs.step(action)
            obs = torch.FloatTensor(next_obs, device=device)

            pbar.refresh()

    def get_dataset(self):
        if hasattr(self, "dataset") and not self.updated:
            return self.dataset
        if self.full:
            pos = self.buffer_size
        else:
            pos = self.pos
        self.dataset = RolloutDataset(
            self.envs, self.obs[:pos], self.action[:pos],
            self.output[:pos], self.value[:pos]
        )
        self.updated = False
        return self.dataset


def gather_data(args, buffer_path, envs, agent, device: torch.device):
    agent.train(False)

    if not isinstance(args, dict):
        args = vars(args)

    # Data gathering
    if not buffer_path.exists():
        # train
        train_b = RolloutBuffer(envs, args["train_bs"], 97)
        train_b.collect_data(agent, None, args["train_collect_samples"],
                             device)
        # val
        val_b = RolloutBuffer(envs, args["val_bs"], 98)
        val_b.collect_data(agent, None, args["val_collect_samples"],
                           device)
        # test
        test_b = RolloutBuffer(envs, args["test_bs"], 99)
        test_b.collect_data(agent, None, args["test_collect_samples"],
                            device)

        with buffer_path.open("wb") as f:
            del train_b.envs
            del val_b.envs
            del test_b.envs
            pickle.dump((train_b, val_b, test_b),
                        f, pickle.HIGHEST_PROTOCOL)
    with buffer_path.open("rb") as f:
        train_b, val_b, test_b = pickle.load(f)
    train_b.envs = deepcopy(envs)
    val_b.envs = deepcopy(envs)
    test_b.envs = deepcopy(envs)

    return train_b, val_b, test_b
