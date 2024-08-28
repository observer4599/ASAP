# The code is from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# python src/algorithm/ppo.py --config-file train_ppo_continuous_action.toml

from typing import Any
from src.utils import (
    get_project_folder, save, parse_args, layer_init)
import tomli
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from tqdm import trange
from src.environment import make_env
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric import Fabric
from pathlib import Path


class Agent(nn.Module):
    def __init__(self, envs, n_units: int = 64, activation_func=nn.Tanh):
        super().__init__()
        assert isinstance(envs.single_action_space,
                          (gym.spaces.Discrete, gym.spaces.Box))
        self.discrete = isinstance(
            envs.single_action_space, gym.spaces.Discrete)

        self.input_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_space = envs.single_action_space
        self.out_dim = self.action_space.n if self.discrete \
            else np.prod(self.action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, n_units)),
            activation_func(),
            layer_init(nn.Linear(n_units, n_units)),
            activation_func(),
            layer_init(nn.Linear(n_units, 1), std=1.0),
        )
        policy = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, n_units)),
            activation_func(),
            layer_init(nn.Linear(n_units, n_units)),
            activation_func(),
            layer_init(nn.Linear(n_units, self.out_dim), std=0.01),
        )
        if self.discrete:
            self.actor = policy
        else:
            self.actor_mean = policy
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.out_dim))

    def forward(self, x, sample: bool = False, raw: bool = False):
        if self.discrete:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if raw:
                output = logits
            else:
                output = probs.probs
        else:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            output = action_mean

        if sample:
            return probs.sample()
        return output

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.discrete:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            output = logits
        else:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            output = action_mean

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        if not self.discrete:
            log_prob = log_prob.sum(1)
            entropy = entropy.sum(1)

        return (action, output, log_prob, entropy, self.critic(x))


def get_args() -> dict[str, Any]:
    with (get_project_folder() / f"configs/{parse_args().config_file}").open("rb") as f:
        args = tomli.load(f)
        args["batch_size"] = int(args["num_envs"] * args["num_steps"])
        args["minibatch_size"] = int(
            args["batch_size"] // args["num_minibatches"])
    return args


if __name__ == "__main__":
    args = get_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = args["torch_deterministic"]

    log_folder = get_project_folder() / f"runs/{args['task']}"
    logger = TensorBoardLogger(root_dir=log_folder,
                               name=f"{args['algorithm']}__{args['env_id']}")
    logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    log_folder = Path(logger.log_dir)

    fabric = Fabric(accelerator=args["accelerator"],
                    loggers=logger)
    fabric.seed_everything(args["seed"], workers=True)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args["env_id"], args["seed"] + i, i, args["capture_video"],
                  args["gamma"], log_folder / "videos", args["record_step"])
         for i in range(args["num_envs"])]
    )

    assert isinstance(envs.single_action_space,
                      (gym.spaces.Discrete, gym.spaces.Box)), \
        f"{envs.single_action_space} action space is not supported"
    discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs)
    optimizer = optim.Adam(
        agent.parameters(), lr=args["learning_rate"], eps=1e-5)
    agent, optimizer = fabric.setup(agent, optimizer)

    # ALGO Logic: Storage setup
    dim = (args["num_steps"], args["num_envs"])
    obs = torch.zeros(
        dim + envs.single_observation_space.shape, device=fabric.device)
    actions = torch.zeros(
        dim + envs.single_action_space.shape, device=fabric.device)
    logprobs = torch.zeros(dim, device=fabric.device)
    rewards = torch.zeros(dim, device=fabric.device)
    dones = torch.zeros(dim, device=fabric.device)
    values = torch.zeros(dim, device=fabric.device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset(seed=args["seed"])[0], dtype=torch.float32,
                            device=fabric.device)
    next_done = torch.zeros(args["num_envs"], device=fabric.device)
    num_updates = args["total_timesteps"] // args["batch_size"]

    for update in (pbar := trange(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        episode_rs = []
        for step in trange(0, args["num_steps"], leave=False):
            global_step += 1 * args["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, device=fabric.device).view(-1)
            next_obs, next_done = torch.tensor(next_obs, dtype=torch.float32,
                                               device=fabric.device), torch.tensor(
                done, device=fabric.device, dtype=torch.float32)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                ep_length = info["episode"]["l"].item()
                ep_return = info["episode"]["r"].item()
                pbar.set_postfix({"episodic_length": ep_length,
                                  "episodic_return": ep_return})
                pbar.refresh()
                fabric.log("charts/episodic_return",
                           ep_return, step=global_step)
                fabric.log("charts/episodic_length",
                           ep_length, step=global_step)
                episode_rs.append(ep_return)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=fabric.device)
            lastgaelam = 0
            for t in reversed(range(args["num_steps"])):
                if t == args["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args["gamma"] * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args["gamma"] * \
                    args["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if discrete:
            b_actions = b_actions.long()

        # Optimizing the policy and value network
        b_inds = np.arange(args["batch_size"])
        clipfracs = []
        for epoch in range(args["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, args["batch_size"], args["minibatch_size"]):
                end = start + args["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args["norm_adv"]:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(
                        ratio, 1 - args["clip_coef"], 1 + args["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args["clip_coef"],
                        args["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args["ent_coef"] * \
                    entropy_loss + v_loss * args["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args["max_grad_norm"])
                optimizer.step()

            if "target_kl" in args.keys():
                if approx_kl > args["target_kl"]:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        fabric.log("charts/learning_rate",
                   optimizer.param_groups[0]["lr"], global_step)
        fabric.log("losses/value_loss", v_loss.item(), global_step)
        fabric.log("losses/policy_loss", pg_loss.item(), global_step)
        fabric.log("losses/entropy", entropy_loss.item(), global_step)
        fabric.log("losses/old_approx_kl",
                   old_approx_kl.item(), global_step)
        fabric.log("losses/approx_kl", approx_kl.item(), global_step)
        fabric.log("losses/clipfrac", np.mean(clipfracs), global_step)
        fabric.log("losses/explained_variance",
                   explained_var, global_step)
        fabric.log("charts/SPS", int(global_step /
                                     (time.time() - start_time)), global_step)
    save(log_folder, envs, agent)
    envs.close()
