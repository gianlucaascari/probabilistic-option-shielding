from .A_star import astar
from .A_star_graph import ASTAR_G

GPlanner_mapper = {
    'A*': astar,
    'A*vg': ASTAR_G
}