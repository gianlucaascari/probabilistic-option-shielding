"""Shielded actor-critic policies.

Both policies reweight the actor's distribution by safety probabilities and
renormalise (PLS). The action-level policy uses ``P(safe | s, a)`` from the
ProbLog shield; the option-level policy uses ``P(safe | s, o)`` from an
option-safety critic (Eq. 2).
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """MLP trunk shared by the actor and the value function."""

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
        )

    def forward(self, observations):
        return self.net(observations)


class _ShieldedPolicy(ActorCriticPolicy):
    """Actor-critic whose action distribution is ``safety * pi / normaliser``."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # The networks are built a second time on purpose: the paper's runs did
        # so, and seeded runs reproduce them only from the RNG state it leaves.
        self._build(lr_schedule)

    def _base_probs(self, obs):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        probs = self._get_action_dist_from_latent(latent_pi).distribution.probs
        return probs, self.value_net(latent_vf)

    def _shielded(self, base_probs, obs, sensor_values=None):
        raise NotImplementedError

    def forward(self, obs, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        base_probs, values = self._base_probs(obs)
        dist = Categorical(probs=self._shielded(base_probs, obs))
        actions = th.argmax(dist.probs, dim=1) if deterministic else dist.sample()
        return actions, values, dist.log_prob(actions)

    def _predict(self, observation, deterministic=False):
        base_probs, _ = self._base_probs(observation)
        dist = Categorical(probs=self._shielded(base_probs, observation))
        return th.argmax(dist.probs, dim=1) if deterministic else dist.sample()

    def evaluate_actions(self, obs, actions, sensor_values=None):
        """Values, log-probabilities and entropy under the shielded policy.

        The unshielded distribution is kept in ``self.last_base_probs`` for the
        safety losses of PPO-PLS and POS-PPO.
        """
        base_probs, values = self._base_probs(obs)
        dist = Categorical(probs=self._shielded(base_probs, obs, sensor_values))
        self.last_base_probs = base_probs
        return values, dist.log_prob(actions), dist.entropy()


class ShieldedActorCriticPolicy(_ShieldedPolicy):
    """Action-level PLS policy (PPO-PLS). Needs ``shield=`` in ``policy_kwargs``."""

    def __init__(self, observation_space, action_space, lr_schedule, shield=None, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.shield = shield

    def _shielded(self, base_probs, obs, sensor_values=None):
        if sensor_values is None:
            sensor_values = self.shield.get_sensor_values()
        # Kept for the safety loss of PPO-PLS, so the shield is evaluated once.
        self.last_action_safeties = self.shield.action_safeties(sensor_values.to(base_probs.device))
        weighted = self.last_action_safeties * base_probs
        return weighted / th.sum(weighted, dim=-1, keepdim=True)


class ShieldedOptionPolicy(_ShieldedPolicy):
    """Option-level policy shielded by ``option_safety(obs) -> P(safe | s, .)``.

    ``option_safety`` is attached by :class:`pos.algorithms.POSPPO` once the
    critic exists.
    """

    option_safety = None

    def _shielded(self, base_probs, obs, sensor_values=None):
        weighted = self.option_safety(obs) * base_probs
        return weighted / (th.sum(weighted, dim=-1, keepdim=True) + 1e-8)
