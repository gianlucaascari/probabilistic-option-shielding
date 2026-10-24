"""PPO-PLS: PPO over primitive actions with a one-step Probabilistic Logic Shield."""
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from ..buffers import SensorRolloutBuffer
from ..policies import ShieldedActorCriticPolicy
from .base import BasePPO


class PPOPLS(BasePPO):
    """PPO whose policy is shielded at every step by ``P(safe | s, a)``.

    The PPO loss gets the safety term ``-alpha * log P_pi(safe | s)`` of
    Yang et al. (2023), where ``P_pi(safe | s) = sum_a P(safe | s, a) pi(a | s)``
    is differentiated through the unshielded policy ``pi``. With ``alpha=0``
    the policy is shielded but trained by the plain PPO loss.

    The sensor readings of every visited state are stored with the rollout,
    so the update re-evaluates the shielded policy on exactly the inputs it
    acted on.
    """

    policy_aliases = {"MlpPolicy": ShieldedActorCriticPolicy}

    def __init__(self, policy, env, shield, alpha=0.0, **kwargs):
        self.shield = shield
        self.alpha = alpha
        policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
        policy_kwargs["shield"] = shield
        kwargs.setdefault("rollout_buffer_class", SensorRolloutBuffer)
        kwargs.setdefault("rollout_buffer_kwargs", {"num_sensors": shield.num_sensors})
        super().__init__(policy, env, policy_kwargs=policy_kwargs, **kwargs)

    def _evaluate_actions(self, batch, actions):
        return self.policy.evaluate_actions(batch.observations, actions, sensor_values=batch.sensor_values)

    def _auxiliary_loss(self, batch):
        if self.alpha == 0:
            return None
        policy_safety = th.sum(self.policy.last_action_safeties * self.policy.last_base_probs, dim=1)
        safety_loss = th.mean(-th.log(policy_safety + 1e-8))
        self._update_logs["train/safety_loss"].append(safety_loss.item())
        return self.alpha * safety_loss

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        """SB3's rollout loop, additionally storing the pre-step sensor readings."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            sensor_values = self.shield.get_sensor_values().detach().cpu().numpy()
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            rollout_buffer.add(self._last_obs, actions.reshape(-1, 1), rewards, self._last_episode_starts,
                               values, log_probs, sensor_values=sensor_values)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True
