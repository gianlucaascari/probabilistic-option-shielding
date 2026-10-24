from .z3 import Z3Planner
from .z3_rl import Z3PlannerRL
from .basic import LocalPlanner

LPlanner_mapper = {
    'Z3_RL': Z3PlannerRL,
    'Z3': Z3Planner,
}
