"""Shared PPO update with extension points."""
from collections import defaultdict

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class BasePPO(PPO):
    """Stable-Baselines3 PPO whose update can be extended by subclasses.

    The update is the standard clipped-surrogate PPO step. Subclasses change
    how actions are evaluated, which advantages enter the surrogate, add a term
    to the loss, or act after each gradient step. Values appended to
    ``self._update_logs[key]`` during the update are logged as their mean.
    """

    def _evaluate_actions(self, batch, actions):
        return self.policy.evaluate_actions(batch.observations, actions)

    def _advantages(self, batch):
        advantages = batch.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _auxiliary_loss(self, batch):
        """Extra term added to the PPO loss, or ``None``."""
        return None

    def _before_update(self):
        pass

    def _after_gradient_step(self, batch):
        pass

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        self._update_logs = defaultdict(list)
        self._before_update()
        logs = self._update_logs
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for batch in self.rollout_buffer.get(self.batch_size):
                actions = batch.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = batch.actions.long().flatten()

                values, log_prob, entropy = self._evaluate_actions(batch, actions)
                values = values.flatten()
                advantages = self._advantages(batch)

                ratio = th.exp(log_prob - batch.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                logs["train/policy_gradient_loss"].append(policy_loss.item())
                logs["train/clip_fraction"].append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.old_values + th.clamp(values - batch.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(batch.returns, values_pred)
                logs["train/value_loss"].append(value_loss.item())

                entropy_loss = -th.mean(entropy)
                logs["train/entropy_loss"].append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                auxiliary = self._auxiliary_loss(batch)
                if auxiliary is not None:
                    loss = loss + auxiliary

                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self._after_gradient_step(batch)

            self._n_updates += 1
            if not continue_training:
                break

        for key, values in logs.items():
            self.logger.record(key, np.mean(values))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
