"""Logging of the metrics reported in the paper."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SafetyMetricsCallback(BaseCallback):
    """Log safety violations and the episodic discounted return.

    Reads ``info["episode"]`` of finished episodes, which must carry
    ``violations`` (a dict rule -> count) and ``r_disc`` (discounted return).
    Logged at the end of every rollout:

    * ``rollout/cum_total_violations``: violations since the start of training;
    * ``rollout/ep_disc_rew_mean``: mean discounted return over the last
      ``stats_window_size`` episodes;
    * ``rollout/ep_total_violations_mean`` and per-rule counterparts.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._cum_episodes = 0
        self._cum_violations = {}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "episode" in info:
                self._cum_episodes += 1
                for rule, count in (info["episode"].get("violations") or {}).items():
                    self._cum_violations[rule] = self._cum_violations.get(rule, 0) + int(count)
        return True

    def _on_rollout_end(self) -> None:
        episodes = self.model.ep_info_buffer
        if not episodes:
            return
        self.logger.record("rollout/ep_disc_rew_mean", float(np.mean([ep["r_disc"] for ep in episodes])))
        per_rule = {}
        for ep in episodes:
            for rule, count in (ep.get("violations") or {}).items():
                per_rule[rule] = per_rule.get(rule, 0) + count
        self.logger.record("rollout/ep_total_violations_mean", sum(per_rule.values()) / len(episodes))
        for rule, count in per_rule.items():
            self.logger.record(f"rollout/ep_violation_{_slug(rule)}_mean", count / len(episodes))
        self.logger.record("rollout/cum_episodes", self._cum_episodes)
        self.logger.record("rollout/cum_total_violations", sum(self._cum_violations.values()))
        for rule, count in self._cum_violations.items():
            self.logger.record(f"rollout/cum_violation_{_slug(rule)}", count)


def _slug(rule):
    return rule.replace(" ", "_").replace("'", "").lower()
