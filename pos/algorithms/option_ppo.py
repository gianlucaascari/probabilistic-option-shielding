"""OptionPPO: PPO over a fixed set of options (Section 2.2)."""
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor

from ..buffers import SMDPRolloutBuffer
from .base import BasePPO


class OptionPPO(BasePPO):
    """PPO whose actions are options.

    The high-level policy picks one of ``num_options`` options; the option's
    intra-option policy then acts until it terminates or the episode ends.
    One rollout entry is one option execution: its reward is the discounted
    reward collected during the option, and its advantage is computed with
    semi-MDP GAE (discount ``gamma ** duration``).

    :param options: the option controller, with methods ``start(obs, option)``,
        ``action(obs, option)`` and ``terminated(obs, option)`` taking a raw
        (not normalised) observation. Defaults to LogiCity's four
        intersection-to-intersection options (``logicity.options``).

    Subclasses hook into the rollout via ``_on_option_start``,
    ``_before_env_step``, ``_on_env_step``, ``_on_option_end``,
    ``_rollout_entry_kwargs`` and ``_on_rollout_end``.
    """

    def __init__(self, policy, env, num_options=4, options=None, **kwargs):
        if options is None:
            from logicity.options import IntersectionOptions
            options = IntersectionOptions.from_env(env)
        self.options = options
        env.action_space = spaces.Discrete(num_options)
        self.num_options = num_options
        kwargs.setdefault("rollout_buffer_class", SMDPRolloutBuffer)
        super().__init__(policy, env, **kwargs)

    def _raw(self, obs):
        """The first raw observation of a batch of policy observations."""
        return self._unnormalize(obs)[0]

    def _unnormalize(self, obs):
        obs = obs.detach().cpu().numpy() if isinstance(obs, th.Tensor) else np.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None, :]
        vec_normalize = self.get_vec_normalize_env()
        return vec_normalize.unnormalize_obs(obs.copy()) if vec_normalize is not None else obs

    # -- rollout hooks -------------------------------------------------------

    def _on_option_start(self, obs_tensor):
        pass

    def _before_env_step(self):
        pass

    def _on_env_step(self, option, action, infos, dones, new_obs, step_in_option):
        pass

    def _on_option_end(self, n_options_collected):
        pass

    def _rollout_entry_kwargs(self):
        return {}

    def _on_rollout_end(self, new_obs, dones):
        pass

    # -- rollout -------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                chosen, values, log_probs = self.policy(obs_tensor)
                self._on_option_start(obs_tensor)
            option = int(chosen.cpu().numpy()[0])
            start_obs = self._last_obs.copy()
            self.options.start(self._raw(start_obs), option)

            option_reward, option_length = 0.0, 0
            while True:
                action = self.options.action(self._raw(self._last_obs), option)
                self._before_env_step()
                new_obs, rewards, dones, infos = env.step(np.array([action]))
                self.num_timesteps += 1
                if dones[0] and "episode" in infos[0]:
                    self.ep_info_buffer.extend([infos[0]["episode"]])
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._on_env_step(option, action, infos, dones, new_obs, option_length)
                option_reward += (self.gamma ** option_length) * rewards[0]
                option_length += 1
                self._last_obs = new_obs
                if dones[0] or self.options.terminated(self._raw(new_obs), option):
                    break

            n_steps += 1
            self._on_option_end(n_steps)
            rollout_buffer.add(start_obs, chosen.cpu().numpy().reshape(-1, 1), np.array([option_reward]),
                               self._last_episode_starts, values, log_probs,
                               option_length=option_length, **self._rollout_entry_kwargs())
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._on_rollout_end(new_obs, dones)
        callback.on_rollout_end()
        return True

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Primitive action for a single environment, re-selecting an option
        whenever the current one has terminated."""
        obs = observation if np.ndim(observation) > 1 else np.asarray(observation)[None, :]
        if episode_start is not None and np.any(episode_start):
            self._current_option = None
        raw = self._raw(obs)
        if getattr(self, "_current_option", None) is None or self.options.terminated(raw, self._current_option):
            option, _ = super().predict(obs, deterministic=deterministic)
            self._current_option = int(option[0])
            self.options.start(raw, self._current_option)
        return np.array([self.options.action(raw, self._current_option)]), None
