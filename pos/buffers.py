"""Rollout buffers.

At the option level one buffer entry is a whole option execution of random
duration ``L``. Returns are therefore discounted by ``gamma ** L`` (semi-MDP
GAE); ``L = 1`` recovers the standard SB3 computation.
"""
from typing import Generator, NamedTuple, Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


def smdp_gae(rewards, values, episode_starts, option_lengths, last_values, dones, gamma, gae_lambda):
    """GAE with the per-entry discount ``gamma ** option_length``."""
    buffer_size = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32)
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]
        discount = gamma ** option_lengths[step]
        delta = rewards[step] + discount * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + discount * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    return advantages


class SMDPRolloutBuffer(RolloutBuffer):
    """Rollout buffer whose entries carry their option duration."""

    def reset(self) -> None:
        self.option_lengths = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, *args, option_length=1.0, **kwargs) -> None:
        self.option_lengths[self.pos] = np.asarray(option_length, dtype=np.float32).reshape(self.n_envs)
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        self.advantages = smdp_gae(
            self.rewards, self.values, self.episode_starts, self.option_lengths,
            last_values.clone().cpu().numpy().flatten(), dones, self.gamma, self.gae_lambda)
        self.returns = self.advantages + self.values


class _ExtendedRolloutBuffer(RolloutBuffer):
    """Rollout buffer that yields extra per-entry arrays alongside the standard ones."""

    extra_fields = ()
    samples_class = None

    def get(self, batch_size: Optional[int] = None) -> Generator[NamedTuple, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns") + self.extra_fields:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size


class SensorRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    sensor_values: th.Tensor


class SensorRolloutBuffer(_ExtendedRolloutBuffer):
    """Also stores the shield's sensor readings of each state, so the PPO
    update can recompute the shielded policy exactly as it was at collection."""

    extra_fields = ("sensor_values",)

    def __init__(self, *args, num_sensors=0, **kwargs):
        self.num_sensors = int(num_sensors)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.sensor_values = np.zeros((self.buffer_size, self.n_envs, self.num_sensors), dtype=np.float32)

    def add(self, *args, sensor_values=None, **kwargs) -> None:
        if sensor_values is not None:
            self.sensor_values[self.pos] = np.asarray(sensor_values, dtype=np.float32).reshape(
                self.n_envs, self.num_sensors)
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds, env=None) -> SensorRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.sensor_values[batch_inds],
        )
        return SensorRolloutBufferSamples(*map(self.to_torch, data))


class CostRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class CostSMDPRolloutBuffer(SMDPRolloutBuffer, _ExtendedRolloutBuffer):
    """SMDP buffer with a parallel cost stream (values, GAE, returns) for the
    Lagrangian baseline. Both streams use the same option-duration discounting."""

    extra_fields = ("cost_values", "cost_advantages", "cost_returns")

    def reset(self) -> None:
        shape = (self.buffer_size, self.n_envs)
        self.costs = np.zeros(shape, dtype=np.float32)
        self.cost_values = np.zeros(shape, dtype=np.float32)
        self.cost_advantages = np.zeros(shape, dtype=np.float32)
        self.cost_returns = np.zeros(shape, dtype=np.float32)
        super().reset()

    def add(self, *args, cost=0.0, cost_value=None, **kwargs) -> None:
        self.costs[self.pos] = np.asarray(cost, dtype=np.float32).reshape(self.n_envs)
        if cost_value is not None:
            self.cost_values[self.pos] = np.asarray(cost_value.detach().cpu().numpy(), dtype=np.float32).reshape(self.n_envs)
        super().add(*args, **kwargs)

    def compute_cost_returns_and_advantage(self, last_cost_values: th.Tensor, dones: np.ndarray) -> None:
        self.cost_advantages = smdp_gae(
            self.costs, self.cost_values, self.episode_starts, self.option_lengths,
            np.asarray(last_cost_values.clone().cpu().numpy(), dtype=np.float32).flatten(),
            dones, self.gamma, self.gae_lambda)
        self.cost_returns = self.cost_advantages + self.cost_values

    def _get_samples(self, batch_inds, env=None) -> CostRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
        )
        return CostRolloutBufferSamples(*map(self.to_torch, data))
