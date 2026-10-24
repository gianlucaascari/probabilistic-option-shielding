"""Sticky-direction inertia (the *Stochastic Map* of the paper).

With probability ``prob`` per step, the agent's chosen action is replaced by
the last non-stop action that was executed. This makes one-step safety
insufficient: the agent commits to a heading and cannot reliably brake at the
last moment.

Semantics:
- The first step after reset never overrides (no prior non-stop action).
- A stop action never updates ``last_action`` (sticky chain survives across stops).
- If the override would push the agent off the agent's ``movable_region``,
  fall back silently to the agent's chosen action.
- ``last_action`` stores what was *executed*, not what was *chosen*.
"""
from __future__ import annotations

import numpy as np
import gym


class InertiaWrapper(gym.Wrapper):
    STOP = 4
    DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

    def __init__(self, env, prob: float, seed: int | None = None):
        super().__init__(env)
        self.prob = float(prob)
        self.rng = np.random.default_rng(seed)
        self.last_action: int | None = None

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

    def seed(self, seed=None):
        """Re-seed the wrapper's inertia RNG and forward to the inner env."""
        self.rng = np.random.default_rng(seed)
        # Forward to the inner env if it supports seed (legacy gym interface).
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]

    def step(self, action):
        chosen = int(np.asarray(action).reshape(-1)[0])
        executed = chosen

        if (
            self.last_action is not None
            and self.last_action != self.STOP
            and self.rng.random() < self.prob
        ):
            dr, dc = self.DELTA[self.last_action]
            r, c = int(self.env.agent.pos[0]), int(self.env.agent.pos[1])
            nr, nc = r + dr, c + dc
            movable = self.env.agent.movable_region
            if (
                0 <= nr < movable.shape[0]
                and 0 <= nc < movable.shape[1]
                and movable[nr, nc] == 1
            ):
                executed = self.last_action

        obs, reward, done, info = self.env.step(executed)
        info = dict(info) if info else {}
        info["inertia_override"] = executed != chosen
        info["inertia_executed_action"] = executed

        if executed != self.STOP:
            self.last_action = executed
        # stop actions leave last_action unchanged

        return obs, reward, done, info

