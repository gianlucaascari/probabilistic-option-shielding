"""OptionPPO-Lag: OptionPPO with a learned Lagrange multiplier (constrained MDP).

The constrained problem ``max E[return] s.t. E[violations per episode] <= d``
is solved through its Lagrangian (Tessler et al. 2019; Ray et al. 2019). PPO
optimises the mixed advantage ``(A_r - lambda A_c) / (1 + lambda)`` and the
multiplier takes one projected gradient step per rollout,

    lambda <- clip(lambda + lambda_lr * (J_cost - d), 0, lambda_max).

The cost of a step is ``info["cost"]`` (one per violated safety rule). The
environment's reward must then carry no safety penalty of its own.
"""
import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from torch.nn import functional as F

from ..buffers import CostSMDPRolloutBuffer
from .option_ppo import OptionPPO


class OptionPPOLagrangian(OptionPPO):
    """
    :param cost_limit: allowed violations per episode ``d``.
    :param lambda_lr: step size of the multiplier.
    :param lambda_init: initial multiplier.
    :param lambda_max: upper bound of the multiplier.
    :param cost_net_arch: hidden sizes of the separate cost value network.
    """

    def __init__(self, policy, env, cost_limit=0.0, lambda_lr=0.035, lambda_init=0.0,
                 lambda_max=100.0, cost_net_arch=(64, 64), **kwargs):
        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.lagrange_lambda = float(lambda_init)
        self.cost_net_arch = tuple(cost_net_arch)
        # Last measured mean episode cost, reused when no episode ends in a rollout.
        self._mean_episode_cost = 0.0
        self._rollout_episode_costs = []
        kwargs.setdefault("rollout_buffer_class", CostSMDPRolloutBuffer)
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        layers, last = [], get_flattened_obs_dim(self.observation_space)
        for hidden in self.cost_net_arch:
            layers += [th.nn.Linear(last, hidden), th.nn.Tanh()]
            last = hidden
        layers.append(th.nn.Linear(last, 1))
        self.cost_value_net = th.nn.Sequential(*layers).to(self.device)
        self.cost_value_optimizer = th.optim.Adam(self.cost_value_net.parameters(), lr=self.lr_schedule(1.0))

    # -- rollout: carry the cost stream alongside the reward -----------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        self._rollout_episode_costs = []
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def _on_option_start(self, obs_tensor):
        self._option_cost_value = self.cost_value_net(obs_tensor).flatten()
        self._option_cost = 0.0

    def _on_env_step(self, option, action, infos, dones, new_obs, step_in_option):
        if dones[0] and "episode" in infos[0]:
            self._rollout_episode_costs.append(float(infos[0]["episode"].get("c", 0.0)))
        self._option_cost += (self.gamma ** step_in_option) * float(infos[0].get("cost", 0.0))

    def _rollout_entry_kwargs(self):
        return {"cost": np.array([self._option_cost]), "cost_value": self._option_cost_value}

    def _on_rollout_end(self, new_obs, dones):
        with th.no_grad():
            last_cost_values = self.cost_value_net(obs_as_tensor(new_obs, self.device)).flatten()
        self.rollout_buffer.compute_cost_returns_and_advantage(last_cost_values, dones)

    # -- update --------------------------------------------------------------

    def _before_update(self):
        """Dual step on the multiplier, using the undiscounted episode cost."""
        if self._rollout_episode_costs:
            self._mean_episode_cost = float(np.mean(self._rollout_episode_costs))
        constraint_gap = self._mean_episode_cost - self.cost_limit
        self.lagrange_lambda = float(
            np.clip(self.lagrange_lambda + self.lambda_lr * constraint_gap, 0.0, self.lambda_max))
        self.logger.record("train/lagrange_lambda", self.lagrange_lambda)
        self.logger.record("train/constraint_gap", constraint_gap)
        self.logger.record("rollout/ep_cost_mean", self._mean_episode_cost)

    def _advantages(self, batch):
        # Both streams are standardised separately before mixing, so lambda
        # does not have to absorb the ratio of their scales.
        advantages, cost_advantages = batch.advantages, batch.cost_advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        lam = self.lagrange_lambda
        return (advantages - lam * cost_advantages) / (1.0 + lam)

    def _after_gradient_step(self, batch):
        cost_values = self.cost_value_net(batch.observations).flatten()
        cost_value_loss = F.mse_loss(batch.cost_returns, cost_values)
        self._update_logs["train/cost_value_loss"].append(cost_value_loss.item())
        self.cost_value_optimizer.zero_grad()
        cost_value_loss.backward()
        th.nn.utils.clip_grad_norm_(self.cost_value_net.parameters(), self.max_grad_norm)
        self.cost_value_optimizer.step()

    def train(self) -> None:
        super().train()
        self.logger.record("train/cost_explained_variance", explained_variance(
            self.rollout_buffer.cost_values.flatten(), self.rollout_buffer.cost_returns.flatten()))
