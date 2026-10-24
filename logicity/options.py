"""The four options of the paper (Appendix B.2, Figure 5).

Each option drives the car from its current position to the next intersection
in one direction: it first circles the closest intersection to the lane that
leaves it in that direction, then drives straight, and terminates when it is
close to the target intersection (or immediately if there is none). Actions
and options are indexed 0 = left, 1 = right, 2 = up, 3 = down; action 4 = stop.
"""
import torch

LEFT, RIGHT, UP, DOWN, STOP = range(5)


def intersections_for_grid(grid_size, spacing=10, offset=1):
    """Intersection centres of a square LogiCity grid."""
    num_streets = (grid_size - 3) // spacing + 1
    return [[offset + r * spacing, offset + c * spacing] for r in range(num_streets) for c in range(num_streets)]


def get_env_intersections(env):
    """Intersections of the city inside (possibly wrapped, vectorised) ``env``."""
    e, seen = env, set()
    while not hasattr(e, "grid_size"):
        if id(e) in seen:
            break
        seen.add(id(e))
        if hasattr(e, "venv"):
            e = e.venv
        elif hasattr(e, "envs"):
            e = e.envs[0]
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    if not hasattr(e, "grid_size"):
        raise ValueError("Cannot find the city grid inside the environment.")
    grid_size = e.grid_size[0]
    return intersections_for_grid(int(grid_size.item() if hasattr(grid_size, "item") else grid_size))


def find_closest_intersection(pos, intersections):
    return min(intersections, key=lambda c: abs(pos[0] - c[0]) + abs(pos[1] - c[1]))


def find_initial_goal(center, direction):
    """Cell from which an option leaves ``center`` in ``direction``."""
    offsets = {LEFT: (-1, -1), RIGHT: (1, 1), UP: (-1, 1), DOWN: (1, -1)}
    dr, dc = offsets[direction]
    return [center[0] + dr, center[1] + dc]


def circular_directions(pos, center):
    """Moves that circle ``center`` counter-clockwise from ``pos``."""
    allowed = torch.zeros(4)
    if pos[0] == center[0] - 1 and pos[1] >= center[1]:
        allowed[LEFT] = True
    if pos[0] == center[0] + 1 and pos[1] <= center[1]:
        allowed[RIGHT] = True
    if pos[1] == center[1] - 1 and pos[0] <= center[0]:
        allowed[DOWN] = True
    if pos[1] == center[1] + 1 and pos[0] >= center[0]:
        allowed[UP] = True
    return allowed


def next_action_to_navigate_intersection(pos, center):
    nz = circular_directions(pos, center).nonzero()
    if nz.numel() > 0:
        return nz[0].item()
    # Off the ring around the centre (e.g. pushed by inertia): step back toward it.
    dr = int(center[0]) - int(pos[0])
    dc = int(center[1]) - int(pos[1])
    if abs(dr) >= abs(dc):
        return UP if dr < 0 else DOWN
    return LEFT if dc < 0 else RIGHT


def action_out_of_intersection(pos, center):
    if pos[0] == center[0] - 1 and pos[1] <= center[1]:
        return LEFT
    if pos[0] == center[0] + 1 and pos[1] >= center[1]:
        return RIGHT
    if pos[1] == center[1] - 1 and pos[0] >= center[0]:
        return DOWN
    if pos[1] == center[1] + 1 and pos[0] <= center[0]:
        return UP


def find_target_intersection(start, option, intersections):
    """Nearest intersection from ``start`` in the option's direction, or None."""
    row, col = start
    if option == LEFT:
        candidates = [i for i in intersections if i[0] == row and i[1] < col]
        return max(candidates, key=lambda x: x[1]) if candidates else None
    if option == RIGHT:
        candidates = [i for i in intersections if i[0] == row and i[1] > col]
        return min(candidates, key=lambda x: x[1]) if candidates else None
    if option == UP:
        candidates = [i for i in intersections if i[1] == col and i[0] < row]
        return max(candidates, key=lambda x: x[0]) if candidates else None
    candidates = [i for i in intersections if i[1] == col and i[0] > row]
    return min(candidates, key=lambda x: x[0]) if candidates else None


def termination(pos, option, start, intersections, threshold=3):
    """Whether an option started at intersection ``start`` has arrived."""
    target = find_target_intersection(start, option, intersections)
    if target is None:
        return True
    if option == LEFT:
        return pos[1] - target[1] < threshold
    if option == RIGHT:
        return target[1] - pos[1] < threshold
    if option == UP:
        return pos[0] - target[0] < threshold
    return target[0] - pos[0] < threshold


def low_level_action_for_option(pos, option, start):
    """Next primitive action of ``option`` started at intersection ``start``.

    Navigation always circles ``start``, never the currently closest
    intersection, so a car pushed off course still leaves on the intended road.
    Once past the exit cell on the exit lane, the option drives straight.
    """
    goal = find_initial_goal(start, option)
    r, c = int(pos[0]), int(pos[1])
    if option == LEFT:
        exited = c < goal[1] and r == goal[0]
    elif option == RIGHT:
        exited = c > goal[1] and r == goal[0]
    elif option == UP:
        exited = r < goal[0] and c == goal[1]
    else:
        exited = r > goal[0] and c == goal[1]
    if exited:
        return int(option)
    if [r, c] == [int(goal[0]), int(goal[1])]:
        return action_out_of_intersection(pos, start)
    return next_action_to_navigate_intersection(pos, start)


class IntersectionOptions:
    """Option controller used by OptionPPO: ``start`` when an option is chosen,
    then ``action`` until ``terminated``. Methods take a single raw
    observation ``[row, col, school_time]``."""

    num_options = 4

    def __init__(self, intersections):
        self.intersections = intersections
        self.start_intersection = None

    @classmethod
    def from_env(cls, env):
        return cls(get_env_intersections(env))

    @staticmethod
    def _position(obs):
        return [int(round(obs[-3])), int(round(obs[-2]))]

    def start(self, obs, option):
        self.start_intersection = find_closest_intersection(self._position(obs), self.intersections)

    def terminated(self, obs, option):
        return termination(self._position(obs), option, self.start_intersection, self.intersections)

    def action(self, obs, option):
        pos = self._position(obs)
        if termination(pos, option, self.start_intersection, self.intersections):
            return STOP
        return low_level_action_for_option(pos, option, self.start_intersection)
