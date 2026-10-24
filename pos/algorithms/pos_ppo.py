"""POS-PPO: PPO over options shielded by a learned option-safety critic."""
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch as th

from ..critics import (
    NeuralSafetyCritic,
    TabularSafetyCritic,
    load_neural_critic_meta,
    neural_td_loss,
    save_neural_critic,
)
from ..policies import ShieldedOptionPolicy
from .option_ppo import OptionPPO


class POSPPO(OptionPPO):
    """Probabilistic Option Shielding on top of OptionPPO (Section 3).

    The option policy ``Pi(o|s)`` is reweighted by ``P(safe | s, o)`` and
    renormalised (Eq. 2), and the PPO loss gets the safety term
    ``-alpha * log P_Pi(safe | s)`` (Yang et al. 2023), where
    ``P_Pi(safe | s) = sum_o P(safe | s, o) Pi(o | s)``.

    The critic is ``"neural"`` or ``"tabular"``. If ``critic_path`` is given it
    is pretrained and frozen; otherwise it is learned online from the one-step
    PLS signal of the ``shield`` on every executed primitive action:

    * tabular: after each option execution, one backward sweep of TD updates
      (Eq. 6) over its transitions, with step size ``tabular_alpha``;
    * neural: every ``critic_update_freq`` option decisions,
      ``critic_grad_steps`` minibatch steps on the TD loss of Eq. (5), sampled
      from a replay buffer of past transitions, with a Polyak target network.

    Option executions cut short by the episode step limit carry no terminal
    safety information and are not used to train the critic.
    """

    policy_aliases = {"MlpPolicy": ShieldedOptionPolicy}

    def __init__(self, policy, env, critic="neural", critic_path=None, shield=None, alpha=0.5,
                 hidden_sizes=(32,), output_power=1, neutral_init=True, critic_lr=1e-3,
                 critic_grad_steps=1, critic_batch_size=128, critic_buffer_size=100_000,
                 critic_update_freq=1, polyak_tau=0.005, table_shape=None, tabular_alpha=0.1,
                 num_options=4, **kwargs):
        self.alpha = alpha
        self.shield = shield
        self.online = critic_path is None
        if self.online and shield is None:
            raise ValueError("Learning the critic online needs a shield providing P(safe | s, a).")
        self.critic_kind = critic
        self._critic_obs_rms = None

        obs_dim = env.observation_space.shape[0]
        device = kwargs.get("device", "cpu")
        if critic == "neural":
            if critic_path is not None:
                meta = load_neural_critic_meta(critic_path)
                hidden_sizes, output_power = meta["hidden_sizes"], meta["output_power"]
                self._critic_obs_rms = meta["obs_rms"]
            self.hidden_sizes = tuple(hidden_sizes)
            self.critic = NeuralSafetyCritic(obs_dim, num_options, hidden_sizes, output_power,
                                             neutral_init=self.online and neutral_init, device=device)
            if critic_path is not None:
                self.critic.load_state_dict(th.load(critic_path, map_location=device))
            self.critic_target = deepcopy(self.critic)
            self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.critic_grad_steps = critic_grad_steps
            self.critic_batch_size = critic_batch_size
            self.critic_update_freq = critic_update_freq
            self.polyak_tau = polyak_tau
            self.replay_buffer = deque(maxlen=critic_buffer_size)
        elif critic == "tabular":
            if critic_path is not None:
                self.critic = TabularSafetyCritic.load(critic_path, device=device)
            else:
                self.critic = TabularSafetyCritic(*table_shape, num_options, alpha=tabular_alpha, device=device)
        else:
            raise ValueError(f"Unknown critic {critic!r}; expected 'neural' or 'tabular'.")
        self._execution = []

        super().__init__(policy, env, num_options=num_options, **kwargs)
        self.policy.option_safety = self.option_safety

    # -- P(safe | s, o) ------------------------------------------------------

    def option_safety(self, obs):
        """``P(safe | s, .)`` for a batch of policy (VecNormalize) observations."""
        with th.no_grad():
            return self.critic(self._critic_input(obs))

    def _critic_input(self, obs):
        """Map policy observations into the space the critic reads.

        * tabular: raw ``[row, col, school_time]``;
        * online neural: the policy observation itself, i.e. normalised with
          the running VecNormalize statistics;
        * pretrained neural: re-normalised with the statistics of the data the
          critic was pretrained on (stored with the checkpoint).
        """
        if self.critic_kind == "tabular":
            return self._unnormalize(obs)
        if self._critic_obs_rms is None:
            return obs
        vec = self.get_vec_normalize_env()
        dev, dtype = obs.device, obs.dtype
        crit_mean = th.as_tensor(self._critic_obs_rms["mean"], device=dev, dtype=dtype)
        crit_std = th.as_tensor(np.sqrt(self._critic_obs_rms["var"] + 1e-8), device=dev, dtype=dtype)
        vec_mean = th.as_tensor(vec.obs_rms.mean, device=dev, dtype=dtype)
        vec_std = th.sqrt(th.as_tensor(vec.obs_rms.var, device=dev, dtype=dtype) + vec.epsilon)
        raw = obs * vec_std + vec_mean
        return th.clamp((raw - crit_mean) / crit_std, -vec.clip_obs, vec.clip_obs)

    # -- safety loss ---------------------------------------------------------

    def _auxiliary_loss(self, batch):
        policy_safety = th.sum(self.policy.last_base_probs * self.option_safety(batch.observations), dim=1)
        safety_loss = th.mean(-th.log(policy_safety.flatten() + 1e-8))
        self._update_logs["train/safety_loss"].append(safety_loss.item())
        return self.alpha * safety_loss

    # -- online critic -------------------------------------------------------

    def _before_env_step(self):
        if self.online:
            self._pre_step_obs = self.get_vec_normalize_env().get_original_obs()[0].copy()
            self._pre_step_sensors = self.shield.get_sensor_values()

    def _on_env_step(self, option, action, infos, dones, new_obs, step_in_option):
        if not self.online:
            return
        # Under inertia the executed action may differ from the chosen one.
        executed = int(infos[0].get("inertia_executed_action", action))
        self._execution.append({
            "unnorm_obs": self._pre_step_obs,
            "unnorm_next_obs": self.get_vec_normalize_env().get_original_obs()[0].copy(),
            "option": option,
            "safe_reward": self.shield.action_safeties(self._pre_step_sensors)[0, executed].item(),
            "next_beta": 1.0 if self.options.terminated(self._raw(new_obs), option) else 0.0,
            "done": dones[0],
            "overtime": bool(infos[0].get("overtime", False)),
        })

    def _on_option_end(self, n_options_collected):
        if not self.online:
            return
        execution, self._execution = self._execution, []
        truncated = execution and execution[-1]["done"] and execution[-1]["overtime"]
        if execution and not truncated:
            if self.critic_kind == "tabular":
                for transition in reversed(self.critic.prepare(execution)):
                    self.critic.update(*transition)
            else:
                self.replay_buffer.extend(execution)
        if self.critic_kind == "neural" and n_options_collected % self.critic_update_freq == 0:
            self._update_neural_critic()

    def _update_neural_critic(self):
        if len(self.replay_buffer) <= self.critic_batch_size:
            return
        buffer = list(self.replay_buffer)
        vec = self.get_vec_normalize_env()
        for _ in range(self.critic_grad_steps):
            sample = random.sample(buffer, self.critic_batch_size)
            batch = (
                vec.normalize_obs(np.asarray([t["unnorm_obs"] for t in sample], dtype=np.float32)),
                np.array([t["option"] for t in sample]),
                vec.normalize_obs(np.asarray([t["unnorm_next_obs"] for t in sample], dtype=np.float32)),
                np.array([t["done"] for t in sample]),
                np.array([t["safe_reward"] for t in sample]),
                np.array([t["next_beta"] for t in sample]),
            )
            loss, _ = neural_td_loss(self.critic, self.critic_target, batch)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            with th.no_grad():
                for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_target.data.mul_(1.0 - self.polyak_tau).add_(self.polyak_tau * p.data)
            self.logger.record_mean("train/critic_loss", loss.item())

    def save_critic(self, path):
        """Save the critic (``.npy`` table, or ``.pt`` + ``.json`` network)."""
        if self.critic_kind == "tabular":
            self.critic.save(path)
        else:
            vec = self.get_vec_normalize_env()
            obs_rms = self._critic_obs_rms
            if obs_rms is None and vec is not None:
                obs_rms = {"mean": vec.obs_rms.mean, "var": vec.obs_rms.var}
            save_neural_critic(self.critic, path, obs_rms=obs_rms, hidden_sizes=self.hidden_sizes)
