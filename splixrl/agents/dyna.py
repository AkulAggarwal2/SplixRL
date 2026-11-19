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

from ..game import N_ACTIONS, PAUSE, STRAIGHT, TURN_LEFT, TURN_RIGHT
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
    # stability options (defaults = the original learner, so old checkpoints load as they were)
    n_step: int = 1  # n-step returns for real data: captures and kills pay off a few steps late
    dueling: bool = False  # separate state value and action advantage (Wang et al. 2016)
    tau: float = 0.0  # > 0: soft target update every step (Polyak); 0: hard copy every target_update
    model_gate: float = 1.0  # plan only while the model's error on changed cells is below this
    n_actions: int = N_ACTIONS  # 4 when the env allows pausing
    # options (SMDP) mode: the world model predicts the state after a whole plan
    # (an option model), so no per-tile view alignment; imagined transitions are
    # discounted by mean_disc, the running mean of gamma ** (plan length)
    smdp: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- networks
class QNetwork(nn.Module):
    def __init__(self, grid_shape: tuple[int, int, int], n_vec: int, hidden: int, dueling: bool = False,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.dueling = dueling
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
        if dueling:
            self.body = nn.Sequential(nn.Linear(n + n_vec, hidden), nn.ReLU())
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Sequential(
                nn.Linear(n + n_vec, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
            )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.conv(grid), vec], 1)
        if not self.dueling:
            return self.head(x)
        h = self.body(x)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(1, keepdim=True)


class WorldModel(nn.Module):
    """(aligned grid, vec, action) -> next-grid logits, next vec, reward, terminal logit."""

    def __init__(self, grid_shape: tuple[int, int, int], n_vec: int, hidden: int, n_actions: int = N_ACTIONS):
        super().__init__()
        self.n_actions = n_actions
        c = grid_shape[0]
        in_c = c + n_actions
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
            nn.Linear(2 * hidden + n_vec + n_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.vec_head = nn.Linear(256, n_vec)
        self.reward_head = nn.Linear(256, 1)
        self.done_head = nn.Linear(256, 1)

    def forward(self, aligned: torch.Tensor, vec: torch.Tensor, action: torch.Tensor):
        a1h = F.one_hot(action, self.n_actions).float()
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
    hold = action == PAUSE
    if hold.any():  # paused: the view stays where it is
        out[hold] = grid[hold]
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
def to_torch(arrays, device) -> tuple[torch.Tensor, ...]:
    """numpy batch -> tensors on the device; actions (int64) stay integers.
    One transfer per tensor: on a GPU each transfer has a fixed cost.

    The copy is synchronous on purpose: the numpy arrays here are temporaries,
    and an asynchronous copy (non_blocking=True) can read them after they are
    freed. That turned a real run's weights into NaN within 20k steps."""
    return tuple(
        torch.as_tensor(a, dtype=torch.int64 if a.dtype == np.int64 else torch.float32).to(device)
        for a in arrays
    )


class ReplayBuffer:
    def __init__(self, capacity: int, grid_shape, n_vec: int, seed: int = 0, gamma: float | None = None):
        self.capacity = capacity
        # per-transition bootstrap discount; set for options (gamma ** option length)
        self.disc = np.full(capacity, np.nan if gamma is None else gamma, np.float32)
        self.store_disc = gamma is not None
        self.grid = np.zeros((capacity, *grid_shape), np.uint8)
        self.vec = np.zeros((capacity, n_vec), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.next_grid = np.zeros((capacity, *grid_shape), np.uint8)
        self.next_vec = np.zeros((capacity, n_vec), np.float32)
        self.done = np.zeros(capacity, np.float32)  # terminated only, never truncated
        self.end = np.zeros(capacity, bool)  # episode ended here (terminated or truncated)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, grid, vec, action, reward, next_grid, next_vec, done, end=None, disc=None) -> None:
        """Add one transition or a batch (leading dim). ``end`` marks the last
        step of an episode (terminated or truncated); defaults to ``done``."""
        grid = np.asarray(grid)
        if grid.ndim == self.grid.ndim - 1:
            grid, vec, next_grid, next_vec = grid[None], np.asarray(vec)[None], np.asarray(next_grid)[None], np.asarray(next_vec)[None]
            action, reward, done = np.array([action]), np.array([reward]), np.array([done])
            end = None if end is None else np.array([end])
        n = len(grid)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.grid[idx] = grid
        self.vec[idx] = vec
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_grid[idx] = next_grid
        self.next_vec[idx] = next_vec
        self.done[idx] = done
        self.end[idx] = np.asarray(done, bool) if end is None else end
        if disc is not None:
            self.disc[idx] = disc
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample_np(self, n: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.size, size=n)
        out = (self.grid[idx], self.vec[idx], self.action[idx], self.reward[idx],
               self.next_grid[idx], self.next_vec[idx], self.done[idx])
        return out + (self.disc[idx],) if self.store_disc else out

    def sample(self, n: int, device) -> tuple[torch.Tensor, ...]:
        return to_torch(self.sample_np(n), device)

    def sample_nstep(self, n: int, steps: int, gamma: float, device) -> tuple[torch.Tensor, ...]:
        return to_torch(self.sample_nstep_np(n, steps, gamma), device)

    def sample_nstep_np(self, n: int, steps: int, gamma: float) -> tuple[np.ndarray, ...]:
        """Batch of n-step transitions built from consecutive single steps.

        Requires transitions added in order from one environment. Returns the
        usual 7 tensors plus the discount to apply to the bootstrap value
        (gamma ** k, where k <= steps is how many steps were summed before the
        episode ended).
        """
        oldest = (self.ptr - self.size) % self.capacity
        span = max(1, self.size - steps + 1)  # starts whose next `steps` entries all exist
        start = (oldest + self.rng.integers(0, span, size=n)) % self.capacity
        ret = np.zeros(n, np.float32)
        disc = np.ones(n, np.float32)
        last = start.copy()
        open_ = np.ones(n, bool)  # still accumulating
        for k in range(steps):
            idx = (start + k) % self.capacity
            ret += open_ * disc * self.reward[idx]
            last = np.where(open_, idx, last)
            disc = np.where(open_, disc * gamma, disc)
            open_ &= ~self.end[idx]
        return (self.grid[start], self.vec[start], self.action[start], ret,
                self.next_grid[last], self.next_vec[last], self.done[last], disc)

    def sample_states_np(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=n)
        return self.grid[idx], self.vec[idx]

    def sample_states(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return to_torch(self.sample_states_np(n), device)


class MultiBuffer:
    """One ReplayBuffer per parallel game, sampled as one. Keeps each game's
    steps contiguous, so n-step returns stay within a game."""

    def __init__(self, n: int, capacity: int, grid_shape, n_vec: int, seed: int = 0, gamma: float | None = None):
        self.bufs = [ReplayBuffer(max(1000, capacity // n), grid_shape, n_vec, seed=seed + i, gamma=gamma)
                     for i in range(n)]
        self.rng = np.random.default_rng(seed + 991)

    @property
    def size(self) -> int:
        return sum(b.size for b in self.bufs)

    def add(self, *args, **kw) -> None:  # single-game use
        self.bufs[0].add(*args, **kw)

    def _split(self, n: int, min_size: int = 1) -> list[int]:
        sizes = np.array([b.size if b.size >= min_size else 0 for b in self.bufs], dtype=float)
        if sizes.sum() == 0:
            raise ValueError("replay buffer is empty")
        return self.rng.multinomial(n, sizes / sizes.sum()).tolist()

    @staticmethod
    def _cat(parts, device):
        # join on the CPU, then move once
        return to_torch(tuple(np.concatenate(xs, 0) for xs in zip(*parts)), device)

    def sample(self, n: int, device):
        return self._cat([b.sample_np(k) for b, k in zip(self.bufs, self._split(n)) if k], device)

    def sample_nstep(self, n: int, steps: int, gamma: float, device):
        return self._cat([b.sample_nstep_np(k, steps, gamma) for b, k in zip(self.bufs, self._split(n, steps)) if k], device)

    def sample_states(self, n: int, device):
        return self._cat([b.sample_states_np(k) for b, k in zip(self.bufs, self._split(n)) if k], device)


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
        self.q = QNetwork(self.grid_shape, n_vec, c.hidden, c.dueling, c.n_actions).to(self.device)
        self.q_target = QNetwork(self.grid_shape, n_vec, c.hidden, c.dueling, c.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=c.lr)
        self.model = WorldModel(self.grid_shape, n_vec, c.model_hidden, c.n_actions).to(self.device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=c.model_lr)
        self.q_updates = 0
        self.mean_disc = self.cfg.gamma ** 10  # updated from real plan lengths in SMDP mode
        self.skipped_updates = 0  # updates dropped because the gradient wasn't finite
        # Acting is one observation at a time, which is slow on a GPU (each call
        # waits for the device). Act with a CPU copy, refreshed every actor_sync updates.
        if self.device.type == "cpu":
            self.q_actor = self.q
        else:
            self.q_actor = QNetwork(self.grid_shape, n_vec, c.hidden, c.dueling, c.n_actions)
            self.sync_actor()

    @property
    def planning(self) -> bool:
        return self.cfg.planning_updates > 0

    # -- acting --------------------------------------------------------------
    def sync_actor(self) -> None:
        if self.q_actor is not self.q:
            self.q_actor.load_state_dict({k: v.detach().cpu() for k, v in self.q.state_dict().items()})

    # Tie-breaking when acting (not in training targets). When moves are nearly
    # tied the argmax flips with tiny noise, which shows up as staircase trails
    # (40% of turns undid the previous turn) and needless pauses. With a margin
    # m: go straight if it is within m of the best move, and only pause when
    # pausing beats every move by more than m.
    act_margin: float = 0.0

    def _pick(self, q: torch.Tensor) -> torch.Tensor:
        best = q.argmax(1)
        m = self.act_margin
        if m <= 0:
            return best
        top = q.gather(1, best[:, None]).squeeze(1)
        choice = torch.where(q[:, STRAIGHT] >= top - m, torch.full_like(best, STRAIGHT), best)
        if q.shape[1] > PAUSE:
            moves = q[:, :PAUSE]
            best_move = moves.argmax(1)
            pause_clear = q[:, PAUSE] > moves.max(1).values + m
            straight_ok = moves[:, STRAIGHT] >= moves.max(1).values - m
            fallback = torch.where(straight_ok, torch.full_like(best, STRAIGHT), best_move)
            choice = torch.where((best == PAUSE) & ~pause_clear, fallback, choice)
        return choice

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True, epsilon: float = 0.0) -> int:
        if not deterministic and self.rng.random() < epsilon:
            return int(self.rng.integers(self.cfg.n_actions))
        grid, vec = obs_to_torch(obs, "cpu")
        return int(self._pick(self.q_actor(grid, vec)).item())

    @torch.no_grad()
    def act_batch(self, obs: dict, epsilon: float = 0.0) -> np.ndarray:
        """Epsilon-greedy actions for a batch of observations (leading dim)."""
        grid = torch.as_tensor(obs["grid"], dtype=torch.float32)
        vec = torch.as_tensor(obs["vec"], dtype=torch.float32)
        actions = self._pick(self.q_actor(grid, vec)).numpy()
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.integers(self.cfg.n_actions, size=int(explore.sum()))
        return actions

    # -- learning --------------------------------------------------------------
    def q_update(self, batch) -> float:
        """Double DQN update. ``batch`` may carry an 8th tensor: per-sample
        bootstrap discount (n-step data); otherwise gamma is used."""
        grid, vec, action, reward, ngrid, nvec, done = batch[:7]
        disc = batch[7] if len(batch) > 7 else self.cfg.gamma
        q = self.q(grid, vec).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            next_a = self.q(ngrid, nvec).argmax(1, keepdim=True)  # online net selects
            next_q = self.q_target(ngrid, nvec).gather(1, next_a).squeeze(1)  # target net evaluates
            target = reward + disc * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        if not torch.isfinite(norm):  # never let one bad batch poison the weights
            self.q_opt.zero_grad(set_to_none=True)
            self.skipped_updates += 1
            return loss.detach()
        self.q_opt.step()
        self.q_updates += 1
        if self.cfg.tau > 0:
            with torch.no_grad():
                for pt, po in zip(self.q_target.parameters(), self.q.parameters()):
                    pt.lerp_(po, self.cfg.tau)
        elif self.q_updates % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        if self.q_updates % self.cfg.actor_sync == 0:
            self.sync_actor()
        return loss.detach()  # a tensor: calling .item() every update would stall the GPU

    def model_update(self, batch, with_metrics: bool = False) -> dict[str, float]:
        grid, vec, action, reward, ngrid, nvec, done = batch[:7]
        logits, pvec, prew, pdone = self.model(self._align(grid, action), vec, action)
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
            changed = (ngrid != self._align(grid, action)) & mask
            changed_err = (pred != ngrid)[changed].float().mean().item() if changed.any() else 0.0
            done_acc = ((pdone > 0).float() == done).float().mean().item()
        return {
            "model_loss": float(loss.item()),
            "model_grid_err": err,  # fraction of cells predicted wrong
            "model_changed_err": changed_err,  # ... among cells that actually changed
            "model_reward_mse": float(rew_loss.item()),
            "model_done_acc": done_acc,
        }

    def _align(self, grid, action):
        return grid if self.cfg.smdp else align(grid, action, self.obs_config)

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
            action = torch.where(explore, torch.randint(0, self.cfg.n_actions, action.shape, device=self.device), action)
            logits, nvec, rew, done_logit = self.model(self._align(grid, action), vec, action)
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
                disc=np.full(len(grid), self.mean_disc, np.float32) if self.cfg.smdp else None,
            )
            added += len(grid)
            keep = done == 0
            grid, vec = ngrid[keep], nvec[keep]
        return added

    # -- persistence -----------------------------------------------------------
    options: dict | None = None  # set for an options agent (see splixrl.options)

    def save(self, path: str, extra: dict | None = None) -> None:
        self.sync_actor()
        if self.options is not None:
            extra = {**(extra or {}), "options": self.options}
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
        agent.options = ck.get("options")
        agent.sync_actor()
        return agent
