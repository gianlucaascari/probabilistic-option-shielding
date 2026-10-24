"""Offline pretraining of option-safety critics (the "pretrained" regime).

Because the option-safety fixed point does not depend on the option-level
policy (Section 3.3), the critic can be fitted on transitions collected by any
policy, e.g. uniformly random options. A transition is a dict as described in
:mod:`pos.critics`; the neural critic additionally reads ``obs`` and
``next_obs``, the observations as normalised during collection.
"""
import random
from copy import deepcopy

import numpy as np
import torch as th
from tqdm import tqdm

from .critics import NeuralSafetyCritic, TabularSafetyCritic, neural_td_loss


def fit_neural_critic(transitions, num_options, hidden_sizes, output_power, lr, gradient_steps,
                      batch_size=128, polyak_tau=0.005, lr_plateau=None, sampling_seed=42,
                      log_interval=100, device="cpu"):
    """Fit a :class:`NeuralSafetyCritic` by minibatch TD learning (Eq. 5).

    Minibatches are drawn uniformly with replacement from ``transitions`` by
    an RNG seeded with ``sampling_seed``; the bootstrap uses a Polyak target.
    ``lr_plateau = {"patience", "factor", "min_lr"}`` halves the step size when
    the mean loss of ``log_interval``-step windows stops improving.
    """
    model = NeuralSafetyCritic(len(transitions[0]["obs"]), num_options, hidden_sizes, output_power, device=device)
    target = deepcopy(model)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_plateau:
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_plateau.get("factor", 0.5),
            patience=lr_plateau["patience"], min_lr=lr_plateau.get("min_lr", 1e-6))

    rng = random.Random(sampling_seed)
    n = len(transitions)
    window_loss = 0.0
    progress = tqdm(range(1, gradient_steps + 1), desc="Fitting neural critic", unit="step")
    for step in progress:
        sample = [transitions[rng.randrange(n)] for _ in range(batch_size)]
        batch = (
            np.array([t["obs"] for t in sample]),
            np.array([t["option"] for t in sample]),
            np.array([t["next_obs"] for t in sample]),
            np.array([t["done"] for t in sample]),
            np.array([t["safe_reward"] for t in sample]),
            np.array([t["next_beta"] for t in sample]),
        )
        loss, _ = neural_td_loss(model, target, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with th.no_grad():
            for p, p_target in zip(model.parameters(), target.parameters()):
                p_target.data.mul_(1.0 - polyak_tau).add_(polyak_tau * p.data)

        window_loss += loss.item()
        if step % log_interval == 0:
            mean_loss, window_loss = window_loss / log_interval, 0.0
            if scheduler is not None:
                scheduler.step(mean_loss)
            progress.set_postfix(loss=f"{mean_loss:.2e}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
    return model


def fit_tabular_critic(transitions, table_shape, num_options, alpha=0.1, max_passes=10_000, tolerance=1e-4,
                       plateau=None):
    """Fit a :class:`TabularSafetyCritic` by repeated TD sweeps (Eq. 6).

    Every pass shuffles the transitions (Python's ``random``) and applies one
    TD update per transition. Training stops once no entry changes by more
    than ``tolerance`` in a pass. ``plateau = {"patience", "factor",
    "min_alpha"}`` multiplies the step size by ``factor`` when that maximum
    change has not improved for ``patience`` passes.
    """
    critic = TabularSafetyCritic(*table_shape, num_options, alpha=alpha)
    prepared = critic.prepare(transitions)
    best_change, stale_passes = float("inf"), 0
    progress = tqdm(range(1, max_passes + 1), desc="Fitting tabular critic", unit="pass")
    for _ in progress:
        random.shuffle(prepared)
        previous = critic.Q.copy()
        for transition in prepared:
            critic.update(*transition)
        max_change = float(np.max(np.abs(critic.Q - previous)))
        progress.set_postfix(max_dQ=f"{max_change:.2e}", alpha=f"{critic.alpha:.1e}")

        if plateau:
            if max_change < best_change:
                best_change, stale_passes = max_change, 0
            else:
                stale_passes += 1
                if stale_passes >= plateau["patience"]:
                    critic.alpha = max(plateau.get("min_alpha", 1e-4), critic.alpha * plateau.get("factor", 0.5))
                    best_change, stale_passes = float("inf"), 0
        if max_change < tolerance:
            break
    return critic
