"""Dyna (Sutton 1991) with a learned neural world model and a Double-DQN learner.

Real experience goes to a replay buffer. From it we train
  * a Q-network (Double DQN target: online net picks a', target net scores it);
  * a world model  (obs, action) -> (next obs, reward, terminal).
Every ``rollout_freq`` steps the model imagines short rollouts
(``rollout_horizon`` steps) from real states, using the current
epsilon-greedy policy; those go to a separate model buffer. After each real
Q-update we do ``planning_updates`` extra Q-updates on imagined batches.
``planning_updates=0`` turns planning (and model training) off, which leaves
the plain Double-DQN learner -- useful as an ablation of what planning adds.

World model and the egocentric view. The local observation is centred on the
player and rotated so it faces up, so after a move the whole picture shifts
by one cell and, after a turn, rotates 90 degrees. That shift/rotation is pure
geometry determined by the action (not game dynamics), and a conv net is bad
at global rotations, so ``align`` applies it to the input first. The model
then only has to predict what changes: the new trail cell, opponent motion,
captures, cells entering the view, reward and death.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game import N_ACTIONS, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ..obs import ObsConfig
from .base import Agent, obs_to_torch


@dataclass
class DynaConfig:
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 2_000
    train_freq: int = 4  # env steps per real Q-update (and per model update)
    target_update: int = 1_000  # Q-updates (real + imagined) between target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # fraction of training over which epsilon decays
    grad_clip: float = 10.0
    hidden: int = 256
    # planning / world model
    planning_updates: int = 3  # imagined Q-updates per real Q-update (0 = no planning)
    model_lr: float = 1e-3
    model_batch_size: int = 64
    model_warmup: int = 5_000  # env steps before imagined data is used
    rollout_freq: int = 250  # env steps between imagination rounds
    rollout_batch: int = 512  # start states per round
    rollout_horizon: int = 1  # imagined steps per start state
    model_buffer_size: int = 20_000
    model_sample: str = "threshold"  # "threshold" (p > .5) or "bernoulli"
    model_hidden: int = 48
    actor_sync: int = 50  # Q-updates between copies of the learner to the CPU acting net

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- networks
class QNetwork(nn.Module):
    def __init__(self, grid_shape: tuple[int, int, int], n_vec: int, hidden: int):
        super().__init__()
        c, h, w = grid_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n + n_vec, hidden), nn.ReLU(), nn.Linear(hidden, N_ACTIONS)
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(grid), vec], 1))


class WorldModel(nn.Module):
    """(aligned grid, vec, action) -> next-grid logits, next vec, reward, terminal logit."""

    def __init__(self, grid_shape: tuple[int, int, int], n_vec: int, hidden: int):
        super().__init__()
        c = grid_shape[0]
        in_c = c + N_ACTIONS
        self.body = nn.Sequential(
            nn.Conv2d(in_c, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        # 1x1 head also sees the input, so "nothing changes" is easy to learn
        self.grid_head = nn.Conv2d(hidden + in_c, c, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden + n_vec + N_ACTIONS, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.vec_head = nn.Linear(256, n_vec)
        self.reward_head = nn.Linear(256, 1)
        self.done_head = nn.Linear(256, 1)

    def forward(self, aligned: torch.Tensor, vec: torch.Tensor, action: torch.Tensor):
        a1h = F.one_hot(action, N_ACTIONS).float()
        planes = a1h[:, :, None, None].expand(-1, -1, aligned.shape[2], aligned.shape[3])
        x = torch.cat([aligned, planes], 1)
        h = self.body(x)
        grid_logits = self.grid_head(torch.cat([h, x], 1))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        z = self.mlp(torch.cat([pooled, vec, a1h], 1))
        return (
            grid_logits,
            vec + self.vec_head(z),  # predict the change in vec
            self.reward_head(z).squeeze(1),
            self.done_head(z).squeeze(1),
        )


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _shift_rotate(grid: torch.Tensor, action: torch.Tensor, shift: bool) -> torch.Tensor:
    out = torch.zeros_like(grid)
    s = action == STRAIGHT
    if s.any():  # new[i, j] = old[i - 1, j]
        if shift:
            out[s, :, 1:, :] = grid[s, :, :-1, :]
        else:
            out[s] = grid[s]
    l = action == TURN_LEFT
    if l.any():  # shift: new[i, j] = old[i, j - 1]; then left -> up (clockwise)
        sh = torch.zeros_like(grid[l]) if shift else grid[l]
        if shift:
            sh[:, :, :, 1:] = grid[l, :, :, :-1]
        out[l] = torch.rot90(sh, k=3, dims=(2, 3))
    r = action == TURN_RIGHT
    if r.any():  # shift: new[i, j] = old[i, j + 1]; then right -> up (counter-clockwise)
        sh = torch.zeros_like(grid[r]) if shift else grid[r]
        if shift:
            sh[:, :, :, :-1] = grid[r, :, :, 1:]
        out[r] = torch.rot90(sh, k=1, dims=(2, 3))
    return out


def align(grid: torch.Tensor, action: torch.Tensor, cfg) -> torch.Tensor:
    """Move the current egocentric view to where the next one will be.

    Fine channels shift one cell in the direction of travel (cells leaving the
    view are dropped, entering cells are 0 = unknown), then rotate so the new
    heading points up. Coarse channels only rotate (a one-tile move is a third
    of a coarse cell); minimap channels are north-up and stay as they are.
    Global views are returned unchanged. ``cfg`` is an ObsConfig (or "local" /
    "global" for the plain six-channel views).
    """
    if isinstance(cfg, str):
        mode, fine, rot = cfg, grid.shape[1], grid.shape[1]
    else:
        mode, fine, rot = cfg.mode, cfg.fine_channels, cfg.rotating_channels
    if mode != "local":
        return grid
    parts = [_shift_rotate(grid[:, :fine], action, shift=True)]
    if rot > fine:
        parts.append(_shift_rotate(grid[:, fine:rot], action, shift=False))
    if grid.shape[1] > rot:
        parts.append(grid[:, rot:])
    return torch.cat(parts, 1) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------- buffers
class ReplayBuffer:
    def __init__(self, capacity: int, grid_shape, n_vec: int, seed: int = 0):
        self.capacity = capacity
        self.grid = np.zeros((capacity, *grid_shape), np.uint8)
        self.vec = np.zeros((capacity, n_vec), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.next_grid = np.zeros((capacity, *grid_shape), np.uint8)
        self.next_vec = np.zeros((capacity, n_vec), np.float32)
        self.done = np.zeros(capacity, np.float32)  # terminated only, never truncated
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, grid, vec, action, reward, next_grid, next_vec, done) -> None:
        """Add one transition or a batch (leading dim)."""
        grid = np.asarray(grid)
        if grid.ndim == self.grid.ndim - 1:
            grid, vec, next_grid, next_vec = grid[None], np.asarray(vec)[None], np.asarray(next_grid)[None], np.asarray(next_vec)[None]
            action, reward, done = np.array([action]), np.array([reward]), np.array([done])
        n = len(grid)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.grid[idx] = grid
        self.vec[idx] = vec
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_grid[idx] = next_grid
        self.next_vec[idx] = next_vec
        self.done[idx] = done
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, device) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, size=n)
        t = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=device)  # noqa: E731
        return (
            t(self.grid),
            t(self.vec),
            t(self.action, torch.int64),
            t(self.reward),
            t(self.next_grid),
            t(self.next_vec),
            t(self.done),
        )

    def sample_states(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=n)
        return (
            torch.as_tensor(self.grid[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.vec[idx], dtype=torch.float32, device=device),
        )


# ---------------------------------------------------------------- agent
class DynaAgent(Agent):
    algo = "dyna"

    def __init__(
        self,
        obs_config: ObsConfig,
        grid_shape: tuple[int, int, int],
        n_vec: int,
        cfg: DynaConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.obs_config = obs_config
        self.grid_shape = tuple(grid_shape)
        self.n_vec = n_vec
        self.cfg = cfg or DynaConfig()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        c = self.cfg
        self.q = QNetwork(self.grid_shape, n_vec, c.hidden).to(self.device)
        self.q_target = QNetwork(self.grid_shape, n_vec, c.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=c.lr)
        self.model = WorldModel(self.grid_shape, n_vec, c.model_hidden).to(self.device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=c.model_lr)
        self.q_updates = 0
        # Acting is one observation at a time, which is slow on a GPU (each call
        # waits for the device). Act with a CPU copy, refreshed every actor_sync updates.
        if self.device.type == "cpu":
            self.q_actor = self.q
        else:
            self.q_actor = QNetwork(self.grid_shape, n_vec, c.hidden)
            self.sync_actor()

    @property
    def planning(self) -> bool:
        return self.cfg.planning_updates > 0

    # -- acting --------------------------------------------------------------
    def sync_actor(self) -> None:
        if self.q_actor is not self.q:
            self.q_actor.load_state_dict({k: v.detach().cpu() for k, v in self.q.state_dict().items()})

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True, epsilon: float = 0.0) -> int:
        if not deterministic and self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        grid, vec = obs_to_torch(obs, "cpu")
        return int(self.q_actor(grid, vec).argmax(1).item())

    # -- learning --------------------------------------------------------------
    def q_update(self, batch) -> float:
        grid, vec, action, reward, ngrid, nvec, done = batch
        q = self.q(grid, vec).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            next_a = self.q(ngrid, nvec).argmax(1, keepdim=True)  # online net selects
            next_q = self.q_target(ngrid, nvec).gather(1, next_a).squeeze(1)  # target net evaluates
            target = reward + self.cfg.gamma * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.q_opt.step()
        self.q_updates += 1
        if self.q_updates % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        if self.q_updates % self.cfg.actor_sync == 0:
            self.sync_actor()
        return loss.detach()  # a tensor: calling .item() every update would stall the GPU

    def model_update(self, batch, with_metrics: bool = False) -> dict[str, float]:
        grid, vec, action, reward, ngrid, nvec, done = batch
        logits, pvec, prew, pdone = self.model(align(grid, action, self.obs_config), vec, action)
        alive = 1.0 - done  # the obs after a death says nothing useful; only learn "done" there
        grid_loss = (F.binary_cross_entropy_with_logits(logits, ngrid, reduction="none").mean((1, 2, 3)) * alive).sum() / alive.sum().clamp(min=1)
        vec_loss = (((pvec - nvec) ** 2).mean(1) * alive).sum() / alive.sum().clamp(min=1)
        rew_loss = F.mse_loss(prew, reward)
        done_loss = F.binary_cross_entropy_with_logits(pdone, done)
        loss = grid_loss + vec_loss + rew_loss + done_loss
        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.model_opt.step()
        if not with_metrics:
            return {}
        with torch.no_grad():
            pred = (logits > 0).float()
            mask = alive[:, None, None, None].bool().expand_as(pred)
            err = (pred != ngrid)[mask].float().mean().item() if mask.any() else 0.0
            changed = (ngrid != align(grid, action, self.obs_config)) & mask
            changed_err = (pred != ngrid)[changed].float().mean().item() if changed.any() else 0.0
            done_acc = ((pdone > 0).float() == done).float().mean().item()
        return {
            "model_loss": float(loss.item()),
            "model_grid_err": err,  # fraction of cells predicted wrong
            "model_changed_err": changed_err,  # ... among cells that actually changed
            "model_reward_mse": float(rew_loss.item()),
            "model_done_acc": done_acc,
        }

    @torch.no_grad()
    def imagine(self, real: ReplayBuffer, imagined: ReplayBuffer, epsilon: float) -> int:
        """Roll the model forward from real states; returns transitions added."""
        c = self.cfg
        grid, vec = real.sample_states(c.rollout_batch, self.device)
        added = 0
        for _ in range(c.rollout_horizon):
            if len(grid) == 0:
                break
            action = self.q(grid, vec).argmax(1)
            explore = torch.as_tensor(self.rng.random(len(grid)) < epsilon, device=self.device)
            action = torch.where(explore, torch.randint(0, N_ACTIONS, action.shape, device=self.device), action)
            logits, nvec, rew, done_logit = self.model(align(grid, action, self.obs_config), vec, action)
            probs = torch.sigmoid(logits)
            ngrid = torch.bernoulli(probs) if c.model_sample == "bernoulli" else (probs > 0.5).float()
            done = (torch.sigmoid(done_logit) > 0.5).float()
            imagined.add(
                grid.to(torch.uint8).cpu().numpy(),
                vec.cpu().numpy(),
                action.cpu().numpy(),
                rew.cpu().numpy(),
                ngrid.to(torch.uint8).cpu().numpy(),
                nvec.cpu().numpy(),
                done.cpu().numpy(),
            )
            added += len(grid)
            keep = done == 0
            grid, vec = ngrid[keep], nvec[keep]
        return added

    # -- persistence -----------------------------------------------------------
    def save(self, path: str, extra: dict | None = None) -> None:
        self.sync_actor()
        torch.save(
            {
                "algo": self.algo,
                "obs_config": self.obs_config.to_dict(),
                "grid_shape": self.grid_shape,
                "n_vec": self.n_vec,
                "config": self.cfg.to_dict(),
                "q": {k: v.detach().cpu() for k, v in self.q.state_dict().items()},
                "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DynaAgent":
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            ObsConfig.from_dict(ck["obs_config"]),
            tuple(ck["grid_shape"]),
            ck["n_vec"],
            DynaConfig(**ck["config"]),
            device=device,
        )
        agent.q.load_state_dict(ck["q"])
        agent.q_target.load_state_dict(ck["q"])
        agent.model.load_state_dict(ck["model"])
        agent.sync_actor()
        return agent
