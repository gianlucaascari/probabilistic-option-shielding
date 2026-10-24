import torch
from queue import PriorityQueue
from ...core.config import *
import numpy as np
import pyastar2d

def astar(movable_map, start, end):
    cost_map = np.zeros(movable_map.shape, dtype=np.float32)
    cost_map[movable_map] = 1
    cost_map[~movable_map] = 1000
    path = pyastar2d.astar_path(cost_map, tuple(start), tuple(end), allow_diagonal=False)
    return torch.tensor(path)