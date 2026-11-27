"""PPO -- Sai's part. Only the interface is here.

What the rest of the project needs from this file:

1. ``PPOAgent(Agent)`` with ``act``, ``save`` and ``load`` (see ``base.py``).
   ``save`` must ``torch.save`` a dict with ``"algo": "ppo"`` and
   ``"obs_config": self.obs_config.to_dict()``; then ``load_agent(path)``,
   ``"frozen:<path>"`` opponents, ``python -m splixrl.evaluate`` and
   ``python -m splixrl.watch`` all work with your checkpoints.
2. Use the exact Round 0 settings from ``splixrl/experiment.py``
   (``round0_env_kwargs()``, ``ROUND0``; reward v2), see docs/TEAM_SETUP.md.
3. A training loop (e.g. ``splixrl/train_ppo.py``) that builds
   ``SplixEnv(**env_kwargs)`` with the same env settings as the other
   algorithms and calls ``TrainingEvaluator(run_dir, env_kwargs,
   DEFAULT_POOL, episodes)(agent, step)`` every N steps, so its eval.csv
   plots next to the others with ``python -m splixrl.plot``.

Observations are dicts: ``obs["grid"]`` float32 (6, 15, 15) and
``obs["vec"]`` float32 (11,). Actions: 0 straight, 1 left, 2 right.
``splixrl.agents.base.obs_to_torch`` batches observations for a network.

Experiment 2 (optional, after the per-tile version works): the same algorithm
over options -- 17 plans (``N_OPTIONS``: rectangular loops, home, reclaim,
attack, ambush, adaptive loops, steal-edge) carried out by scripted controllers (``splixrl/options.py``).

* Env: ``splixrl.options.make_options_env(gamma=..., **env_kwargs)``, a normal
  Gymnasium env with ``Discrete(N_OPTIONS)`` actions. One ``step`` runs a whole
  plan and returns the discounted reward collected during it.
* Plans take different numbers of tiles. ``info["option_disc"]`` is gamma ** k
  for a k-tile plan: use it instead of gamma when bootstrapping
  (target = r + option_disc * (1 - done) * Q(s', a')).
  For PPO the same goes inside GAE: delta = r + option_disc * V(s') - V(s),
  and the advantage recursion uses option_disc * lambda in place of gamma * lambda.
* Observations at decision points are the same dicts as above (use the same
  observation settings as the Dyna options run, see README "Experiment 2").
* Save ``"options": {"safety": 6.0, "lag_slack": 30.0}`` in the checkpoint;
  ``load_agent`` then marks the agent as an options agent and every tool plays
  it plan by plan. ``act`` returns an option index in ``range(N_OPTIONS)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


import numpy as np 
import torch
import torch.nn as nn 
from torch.distributions import Categorical

from ..game import N_ACTIONS
from ..obs import ObsConfig
from .base import Agent, obs_to_torch


@dataclass
class PPOConfig:
    gamma: float = 0.99            # must match ROUND0["gamma"]
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    anneal_lr: bool = True         # linearly decay lr to 0 over training
    num_envs: int = 8              # parallel games per rollout
    rollout_steps: int = 128       # steps per game per rollout -> batch = 8 * 128 = 1024
    epochs: int = 4                # passes over each batch
    minibatch_size: int = 256
    clip_coef: float = 0.2         # PPO ratio clip
    clip_vloss: bool = False       # value clipping; off because returns are ~tens, not ~1
    ent_coef: float = 0.01         # entropy bonus (exploration)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.0         # > 0: stop an update's epochs early if KL exceeds this
    norm_adv: bool = True          # normalize advantages per minibatch
    hidden: int = 256
    n_actions: int = N_ACTIONS     # evaluate() reads this to decide on pausing

    def to_dict(self) -> dict:
        return asdict(self)

def layer_init(layer: nn.Linear | nn.Conv2d, std: float = float(np.sqrt(2)), bias: float = 0.0) -> nn.Linear | nn.Conv2d:
    "Orthagonal init, the standard choice for PPO"

    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None: 
      nn.init.constant_(layer.bias, bias) 
    return layer

class ActorCritic(nn.Module): 
    "grid -> conv turnk"
    "concat vec -> hidden -> (policy logits, value)"

    def __init__(self, grid_shape: tuple[int, int, int], n_vec: int, hidden: int, n_actions: int):
        super().__init__()
        c, h, w = grid_shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad(): 
            n = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.body = nn.Sequential(layer_init(nn.Linear(n + n_vec, hidden)), nn.ReLU())
        self.policy = layer_init(nn.Linear(hidden, n_actions), std=0.01) # near-uniform at the start 
        self.value = layer_init(nn.Linear(hidden, 1), std=1.0)
    
    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        h = self.body(torch.cat([self.conv(grid), vec], 1))
        return self.policy(h), self.value(h).squeez(-1)

class PPOAgent(Agent):
    algo = "ppo"
    options: dict | None = None # is set only for an options agent (Experiemtn 2)

    def __init__(self, obs_config: ObsConfig, grid_shape, n_vec: int,
                 cfg: PPOConfig | None = None, device: str = "cpu", seed: int = 0):
        self.obs_config = obs_config
        self.grid_shape = tuple(grid_shape)
        self.n_vec = n_vec
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        c = self.cfg 
        self.net = ActorCritic(self.grid_shape, n_vec, c.hidden, c.n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr, eps=1e-5)

        
        # acts with a cpu copy, refreshed after every update 
        if self.device.type == "cpu": 
            self.net_cpu = self.net
        else: 
            self.net_cpu = ActorCritic(self.grid_shape, n_vec, c.hidden, c.n_actions)
            self.sync_actor()
    
    def sync_actor(self) -> None: 
        if self.net_cpu is not self.net: 
          self.net_cpu.load_state_dict({k: v.detach().cpu() for k, v in self.net.state_dict().items()})


    # ACTING
    @torch.no_grad()
    def act(self, obs, deterministic: bool = True) -> int: 
        "evaluation, watch and frozen oppoents: one obs on cpu"
        grid, vec = obs_to_torch(obs, "cpu")
        logits, _ = self.net_cpu(grid, vec) 
        if deterministic: 
            return int(logits.max(1).item())
        return int(Categorical(logits=logits).sample().item())

    # LEARNING
    def update(self, batch: dict[str, np.ndarray], lr: float) -> dict[str, float]:
        "PPO update on one rollout"
        "batch holds flat arrays: grid, vec, actions, logprobs, advantages, returns, values (one row per transition)"
        c = self.cfg
        for g in self.opt.param_groups:
            g["lr"] = lr 
        t = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        n = len(t["actions"])
        stats: dict[str, list[float]] = defaultdict(list)
        for _ in range(c.epochs): 
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, c.minibatch_size):
                idx = perm[start:start + c.minibatch_size]
                logits, value = self.net(t["grid"][idx], t["vec"][idx])
                dist = Categorical(logits=logits)
                log_ratio = dist.log_prob(t["actions"][idx] - t["logprobs"][idx])
                ratio = log_ratio.exp()

                adv = t["advantages"][idx]
                if c.norm_adv: 
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()


                ret = t["returns"][idx]
                if c.clip_vloss:
                    old_v = t["values"][idx]
                    v_clip = old_v + (value - old_v).clamp(-c.clip_coef, c.clip_coef)
                    v_loss = 0.5 * torch.max((value - ret) ** 2, (v_clip - ret) ** 2).mean()
                else:
                    v_loss = 0.5 * ((value - ret) ** 2).mean()

                entropy = dist.entropy().mean()
                loss = pg_loss + c.vf_coef * v_loss - c.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                with torch.no_grad(): 
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > c.clip_coef).float().mean().item()
                stats["policy_loss"].append(pg_loss.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)
                stats["clipfrac"].append(clipfrac)
                if c.target_kl > 0 and approx_kl > c.target_kl: 
                    break
        self.sync_actor()
        return {k: float(np.mean(v)) for k, v in stats.items()}
  
    # CHECKPOINTS
    def save(self, path: str, extra: dict | None = None) -> None:
        extra = dict(extra or {})
        if self.options is not None:
            extra["options"] = self.options

        torch.save({
            "algo": self.algo, 
            "obs_config": self.obs_config.to_dict(), 
            "grid_shape": self.grid_shape, 
            "n_vec": self.n_vec, 
            "config": self.cfg.to_dict(),
            "net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            **extra,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPOAgent":
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ObsConfig.from_dict(ck["obs_config"]), tuple(ck["grid_shape"]), ck["n_vec"],
                    PPOConfig(**ck["config"]), device=device)
        agent.net.load_state_dict(ck["net"])
        agent.sync_actor()
        agent.options = ck.get("options")
        return agent


# Note for PPO: ``act(obs, deterministic=True)`` should return the most likely
# action (used by evaluation and frozen opponents); sample when
# deterministic=False. gymnasium.vector.SyncVectorEnv works with SplixEnv if
# you want several environments per update.
