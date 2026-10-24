"""Option-safety critics estimating ``P(safe | s, o)`` (Section 3.3).

Both critics are trained toward the bootstrapped TD target of Eq. (5),

    y_t = P(safe | s_t, a_t) * [beta(s_{t+1}) + (1 - beta(s_{t+1})) P(safe | s_{t+1}, o)],

where ``P(safe | s_t, a_t)`` is the one-step PLS signal of the executed action
and ``beta`` the termination indicator of the option. The end of an episode
also ends the option.

A transition is a dict with keys ``unnorm_obs``, ``unnorm_next_obs`` (raw
``[row, col, school_time]`` observations), ``option``, ``safe_reward``
(the PLS probability), ``next_beta`` and ``done``.
"""
import json
import os

import numpy as np
import torch as th
import torch.nn as nn


class NeuralSafetyCritic(nn.Module):
    """MLP critic with output ``P(safe | s, o) = sigmoid(z_o) ** k``.

    The power ``k`` (``output_power``) lets very small probabilities be
    represented without extreme logits (Appendix B.5).

    :param neutral_init: zero the output layer so every option starts at
        ``sigmoid(0) ** k`` instead of a random per-option bias.
    """

    def __init__(self, obs_dim, num_options, hidden_sizes=(32,), output_power=1,
                 neutral_init=False, device="cpu"):
        super().__init__()
        self.device = device
        self.output_power = int(output_power)
        layers, prev = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_options))
        self.net = nn.Sequential(*layers)
        if neutral_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, 0.0)
        self.to(device)

    def forward_root(self, obs):
        """``sigmoid(z)``, the ``k``-th root of the predicted probability."""
        if isinstance(obs, np.ndarray):
            obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        return th.sigmoid(self.net(obs))

    def forward(self, obs):
        root = self.forward_root(obs)
        return root if self.output_power == 1 else th.pow(root, self.output_power)


def neural_td_loss(model, target_model, batch):
    """Mean squared TD error of Eq. (5) for a minibatch.

    ``batch`` is ``(obs, options, next_obs, dones, safe_rewards, next_betas)``
    with observations already in the critic's input space. The bootstrap uses
    the Polyak-averaged ``target_model``. With ``output_power = k`` the fit is
    done on the ``k``-th root of both sides; the recursion is multiplicative,
    so it is closed under the root and the fixed point is unchanged.
    """
    obs, options, next_obs, dones, safe_rewards, next_betas = batch
    device = model.device
    obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
    next_obs_t = th.as_tensor(next_obs, dtype=th.float32, device=device)
    options_t = th.as_tensor(options, dtype=th.long, device=device)
    dones_t = th.as_tensor(dones, dtype=th.float32, device=device)
    safe_rew_t = th.as_tensor(safe_rewards, dtype=th.float32, device=device)
    next_betas_t = th.as_tensor(next_betas, dtype=th.float32, device=device)
    idx = th.arange(len(options_t), device=device)

    with th.no_grad():
        # continues = 1 when the option keeps running from s_{t+1}, else 0.
        continues = (1.0 - next_betas_t) * (1.0 - dones_t)
        next_v = target_model(next_obs_t)[idx, options_t]
        target = safe_rew_t * th.pow(next_v.clamp(min=1e-8), continues)

    k = model.output_power
    if k == 1:
        pred = model(obs_t)[idx, options_t]
        loss = nn.functional.mse_loss(pred, target)
    else:
        root = model.forward_root(obs_t)[idx, options_t]
        loss = nn.functional.mse_loss(root, th.pow(target.clamp(min=0.0), 1.0 / k))
        pred = th.pow(root, k)
    return loss, (pred - target).abs().mean().item()


def save_neural_critic(model, path, obs_rms=None, hidden_sizes=None):
    """Write ``<path>.pt`` (state dict) and ``<path>.json`` (architecture and
    the observation statistics the critic was trained under)."""
    base = path[:-3] if path.endswith(".pt") else path
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    th.save(model.state_dict(), base + ".pt")
    meta = {
        "hidden_sizes": list(hidden_sizes),
        "output_power": model.output_power,
        "num_options": model.net[-1].out_features,
        "obs_dim": model.net[0].in_features,
        "obs_rms": None if obs_rms is None else {
            "mean": np.asarray(obs_rms["mean"]).tolist(),
            "var": np.asarray(obs_rms["var"]).tolist(),
        },
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_neural_critic_meta(path):
    base = path[:-3] if path.endswith(".pt") else path
    with open(base + ".json") as f:
        meta = json.load(f)
    if meta.get("obs_rms") is not None:
        meta["obs_rms"] = {k: np.asarray(v, dtype=np.float32) for k, v in meta["obs_rms"].items()}
    return meta


class TabularSafetyCritic:
    """Table ``Q[row, col, school_time, option]`` of option-safety probabilities.

    Entries start at 0.5 and are updated with step size ``alpha`` toward the
    TD target (Eq. 6). Lookups take raw observations.
    """

    def __init__(self, grid_h, grid_w, num_time_slots, num_options, alpha=0.1, device="cpu"):
        self.Q = np.full((grid_h, grid_w, num_time_slots, num_options), 0.5, dtype=np.float32)
        self.alpha = alpha
        self.device = device

    def _state(self, obs):
        obs = np.asarray(obs, dtype=np.float32).flatten()
        h, w, t_slots, _ = self.Q.shape
        r = min(max(int(round(float(obs[-3]))), 0), h - 1)
        c = min(max(int(round(float(obs[-2]))), 0), w - 1)
        t = min(max(int(round(float(obs[-1]))), 0), t_slots - 1)
        return r, c, t

    def __call__(self, obs):
        """(batch, obs_dim) raw observations -> (batch, num_options) tensor."""
        obs = obs.detach().cpu().numpy() if isinstance(obs, th.Tensor) else np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        out = np.stack([self.Q[self._state(o)].copy() for o in obs])
        np.maximum(out, 1e-8, out=out)
        return th.tensor(out, dtype=th.float32, device=self.device)

    def prepare(self, transitions):
        """Pre-compute table indices: tuples ``(s, option, safe_reward, s', continues)``."""
        return [
            (self._state(tr["unnorm_obs"]), int(tr["option"]), float(tr["safe_reward"]),
             self._state(tr["unnorm_next_obs"]),
             float(tr["next_beta"]) < 0.5 and float(tr["done"]) < 0.5)
            for tr in transitions
        ]

    def update(self, s, option, safe_reward, next_s, continues):
        """One TD step (Eq. 6) on a prepared transition. Returns the change."""
        target = safe_reward * float(self.Q[next_s + (option,)]) if continues else safe_reward
        target = min(max(target, 0.0), 1.0)
        old = float(self.Q[s + (option,)])
        self.Q[s + (option,)] = min(max(old + self.alpha * (target - old), 0.0), 1.0)
        return abs(float(self.Q[s + (option,)]) - old)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.Q)

    @classmethod
    def load(cls, path, device="cpu"):
        Q = np.load(path)
        critic = cls(*Q.shape, device=device)
        critic.Q = Q
        return critic
