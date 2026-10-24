"""Random start-position sampling for HierarchicalCar.

On every reset, samples a start cell from a precomputed set of "expected
option endpoints" — the positions where a deterministic option execution
naturally terminates on the drivable region. Used when collecting
data to pretrain an option-safety critic, so that the state distribution
matches the states in which an option-level policy makes decisions.

Endpoint geometry per terminating intersection B, following
logicity.options.find_initial_goal (travel-lane offset ±1) and
termination (threshold=3, i.e. stops 2 cells before target center):

  Option 0 (LEFT)  arriving at B: (r_B - 1, c_B + 2)
  Option 1 (RIGHT) arriving at B: (r_B + 1, c_B - 2)
  Option 2 (UP)    arriving at B: (r_B + 2, c_B + 1)
  Option 3 (DOWN)  arriving at B: (r_B - 2, c_B - 1)

Endpoints falling outside the drivable region (Traffic Street ∪ Overlap) —
e.g. off-grid at edge intersections — are dropped.

Sits between GymCityWrapper and InertiaWrapper: the position must be set
before GymCityWrapper.reset runs HierarchicalCar.init.
"""
from __future__ import annotations

import gym
import numpy as np

from ..core.config import STREET_ID, LABEL_MAP
from ..options import intersections_for_grid


_TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}
_TRAFFIC_STREET = _TYPE_MAP["Traffic Street"]
_CROSSING_STREET = _TYPE_MAP["Overlap"]

# (row_offset, col_offset) relative to the terminating intersection center,
# indexed by incoming option direction.
_ENDPOINT_OFFSETS = [
    (-1, +2),  # 0 LEFT
    (+1, -2),  # 1 RIGHT
    (+2, +1),  # 2 UP
    (-2, -1),  # 3 DOWN
]


def _compute_option_endpoints(city_grid, grid_size):
    street_layer = city_grid[STREET_ID]
    if hasattr(street_layer, "numpy"):
        street_layer = street_layer.numpy()
    drivable = (street_layer == _TRAFFIC_STREET) | (street_layer == _CROSSING_STREET)
    H, W = drivable.shape

    endpoints = []
    seen = set()
    for r_i, c_i in intersections_for_grid(int(grid_size)):
        for dr, dc in _ENDPOINT_OFFSETS:
            r, c = r_i + dr, c_i + dc
            if 0 <= r < H and 0 <= c < W and drivable[r, c] and (r, c) not in seen:
                endpoints.append((int(r), int(c)))
                seen.add((r, c))
    return endpoints


class RandomStartWrapper(gym.Wrapper):
    """On reset, sample a start position from the precomputed endpoint set."""

    def __init__(self, env, seed=None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        grid_size = env.env.grid_size[0]
        if hasattr(grid_size, "item"):
            grid_size = grid_size.item()
        self.valid_starts = _compute_option_endpoints(env.env.city_grid, int(grid_size))
        if not self.valid_starts:
            raise RuntimeError(
                "RandomStartWrapper: no valid option-endpoint starts found for this map."
            )

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        r, c = self.valid_starts[int(self.rng.integers(len(self.valid_starts)))]
        self.env.agent.hierarchical_task["start"] = [r, c]
        return self.env.reset(**kwargs)

