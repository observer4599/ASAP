# The use of wrappers is learned from CleanRL

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.renderer import FlappyBirdRenderer
import pygame
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from gymnasium.experimental.wrappers import RecordVideoV0


class ConverterWrapper(gym.Wrapper):
    def __init__(self, env, env_id: str):
        super().__init__(env)
        self.env_id = env_id

        np.random.seed(0)
        if env_id == "FlappyBird-v0":
            self.base_state = np.array([[288., 0., 512.,
                                         288., 0., 512.,
                                         288., 0., 512.,
                                         244., 0.,  45.]])
        elif env_id == "Acrobot-v1":
            self.base_state = np.array(
                [np.cos(0.0), np.sin(0.0), np.cos(0.0), np.sin(0.0), 0.0, 0.0])
        elif env_id == "Pendulum-v1":
            self.base_state = np.array(
                [np.cos(np.pi), np.sin(np.pi), 0.0])
        elif env_id == "MountainCar-v0":
            self.base_state = np.array([-0.5, 0])
        elif env_id == "CartPole-v1":
            self.base_state = np.zeros((1, 4))
        elif env_id == "Swimmer-v4":
            self.base_state = np.zeros((1, 8))
        else:
            raise NotImplementedError(f"Not implemented for {self.env_id}")

    def get_normalized_obs(self, obs):
        mean = self.env.obs_rms.mean[np.newaxis, ...]
        var = self.env.obs_rms.var[np.newaxis, ...]
        epsilon = self.env.epsilon

        obs = obs.copy() - mean
        return obs / np.sqrt(var + epsilon)

    def get_unnormalized_obs(self, obs):
        mean = self.env.obs_rms.mean[np.newaxis, ...]
        var = self.env.obs_rms.var[np.newaxis, ...]
        epsilon = self.env.epsilon

        obs = obs.copy() * np.sqrt(var + epsilon)
        return obs + mean


class ExtendedFlappyBirdEnv(flappy_bird_gymnasium.FlappyBirdEnvSimple):
    def __init__(self, screen_size: tuple[int, int] = (288, 512),
                 audio_on: bool = True,
                 normalize_obs: bool = True, pipe_gap: int = 100,
                 bird_color: str = "yellow", pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        super().__init__(screen_size, audio_on, normalize_obs,
                         pipe_gap, bird_color, pipe_color, background)
        self.render_mode = "rgb_array"
        self.id = "FlappyBird-v0"

    def render(self, show_score: bool = False):
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(
                screen_size=self._screen_size,
                audio_on=self._audio_on,
                bird_color=self._bird_color,
                pipe_color=self._pipe_color,
                background=self._bg_type,
            )
            self._renderer.game = self._game
        self._renderer.draw_surface(show_score=show_score)
        return np.swapaxes(pygame.surfarray.array3d(self._renderer.surface),
                           0, 1)


class ExtendedNormalizeObservation(NormalizeObservation):
    def __init__(self, env, epsilon: float = 1e-8, normalize_obs: bool = True):
        super().__init__(env, epsilon)
        self.normalize_obs = normalize_obs

    def normalize(self, obs):
        if self.normalize_obs:
            return super().normalize(obs)
        return (obs - self.obs_rms.mean) \
            / np.sqrt(self.obs_rms.var + self.epsilon)


class ExtendedNormalizeReward(NormalizeReward):
    def __init__(self, env, gamma: float = 0.99, epsilon: float = 1e-8,
                 normalize_rews: bool = True):
        super().__init__(env, gamma, epsilon)
        self.normalize_rews = normalize_rews

    def normalize(self, rews):
        if self.normalize_rews:
            return super().normalize(rews)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


ENV_PPO_CONTINUOUS = (
    "Swimmer-v4",)
SMALL_ENV = ("CartPole-v1", "MountainCar-v0", "Acrobot-v1",)
LEGAL_ENV = SMALL_ENV + ENV_PPO_CONTINUOUS


def seed_env(env: gym.Env, seed: int) -> None:
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def make_env(env_id: str, seed: int, idx: int, capture_video: bool,
             gamma: float, run_folder: Optional[Path] = None,
             record_step: int = 20_000, normalize: bool = True) -> Callable[[], gym.Env]:

    def thunk_small() -> gym.Env:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, run_folder)
        env = ExtendedNormalizeObservation(env, normalize_obs=normalize)
        env = ExtendedNormalizeReward(
            env, gamma=gamma, normalize_rews=normalize)
        seed_env(env, seed)

        env = ConverterWrapper(env, env_id)

        return env

    def thunk_ppo_continuous():
        env = gym.make(env_id, render_mode="rgb_array")
        # deal with dm_control's Dict observation space
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = RecordVideoV0(env, run_folder, video_length=1024)
        env = gym.wrappers.ClipAction(env)
        env = ExtendedNormalizeObservation(env, normalize_obs=normalize)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = ExtendedNormalizeReward(
            env, gamma=gamma, normalize_rews=normalize)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10))

        env = ConverterWrapper(env, env_id)
        seed_env(env, seed)

        return env

    def thunk_flappybird():
        env = ExtendedFlappyBirdEnv(audio_on=False, normalize_obs=False)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=4096)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, run_folder, video_length=1000)
        env = ExtendedNormalizeObservation(env, normalize_obs=normalize)
        env = ConverterWrapper(env, env_id)
        return env

    if env_id in SMALL_ENV:
        return thunk_small
    elif env_id in ENV_PPO_CONTINUOUS:
        return thunk_ppo_continuous
    elif env_id == "FlappyBird-v0":
        return thunk_flappybird
    raise NotImplementedError
