from .basic import Agent
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from ..utils.gen import gen_occ
from ..utils.find import find_nearest_building, find_building_mask
from ..utils.sample import sample_start_goal, sample_start_goal_vh, sample_determine_start_goal
from ..planners import GPlanner_mapper
from ..core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class HierarchicalCar(Agent):
    def __init__(self, size, id, world_state_matrix, global_planner, concepts, hierarchical_task, init_info=None, debug=False, region=240, device="cpu"):
        self.device=device
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner_type = global_planner
        self.hierarchical_task = hierarchical_task
        super().__init__(size, id, world_state_matrix, concepts, init_info=init_info, debug=debug, region=region)
        # Actions: ["left_1", "right_1", "up_1", "down_1", "left_2", "right_2", "up_2", "down_2", "left_3", "right_3", "up_3", "down_3", "stop"]
        self.action_space = torch.tensor(range(13))
        self.action_to_move = {
            self.action_space[0].item(): torch.tensor((0, -1)),
            self.action_space[1].item(): torch.tensor((0, 1)),
            self.action_space[2].item(): torch.tensor((-1, 0)),
            self.action_space[3].item(): torch.tensor((1, 0)),
            self.action_space[4].item(): torch.tensor((0, -2)),
            self.action_space[5].item(): torch.tensor((0, 2)),
            self.action_space[6].item(): torch.tensor((-2, 0)),
            self.action_space[7].item(): torch.tensor((2, 0)),
            self.action_space[8].item(): torch.tensor((0, -3)),
            self.action_space[9].item(): torch.tensor((0, 3)),
            self.action_space[10].item(): torch.tensor((-3, 0)),
            self.action_space[11].item(): torch.tensor((3, 0))
        }
        self.move_to_action = {
            self.action_to_move[k]: k for k in self.action_to_move
        }
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_Slow", 
            1: "Right_Slow", 
            2: "Up_Slow", 
            3: "Down_Slow", 
            4: "Left_Normal", 
            5: "Right_Normal", 
            6: "Up_Normal", 
            7: "Down_Normal",
            8: "Left_Fast", 
            9: "Right_Fast", 
            10: "Up_Fast", 
            11: "Down_Fast",
            12: "Stop"
            }

    def init(self, world_state_matrix, init_info=None, debug=False):
        Traffic_STREET = TYPE_MAP['Traffic Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        if init_info is not None:
            self.init_from_dict(init_info)
        else:
            self.start = torch.tensor(self.hierarchical_task['start'])
            self.goal = torch.tensor(self.hierarchical_task['goal'])
            self.pos = self.start.clone()
            
        # specify the occupacy map
        self.midline_matrix = (world_state_matrix[STREET_ID] == Traffic_STREET+MID_LINE_CODE_PLUS)
        self.movable_region = (world_state_matrix[STREET_ID] == Traffic_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET) | self.midline_matrix
        self.global_planner = GPlanner_mapper[self.global_planner_type](self.movable_region, self.midline_matrix, CAR_STREET_OFFSET)
        
        points = []
        if len(self.global_planner.start_lists) > 0:
            points.append(torch.cat(self.global_planner.start_lists, dim=0))
        if len(self.global_planner.end_lists) > 0:
            points.append(torch.cat(self.global_planner.end_lists, dim=0))
            
        if len(points) > 0:
            self.intersection_points = torch.cat(points, dim=0)
        else:
            self.intersection_points = torch.tensor([])

        # get global traj on the occupacy map
        self.reach_goal = False
        self.last_move_dir = None
        logger.debug("{}_{} initialization done!".format(self.type, self.id))

    def reset_concepts(self, max_priority, concepts_dist=None):
        self.priority = np.random.randint(1, max_priority)
        if concepts_dist is not None:
            self.concepts = {
                "type": "Car"
            }
            sample = random.choices(concepts_dist['concepts'], weights=concepts_dist['prob'], k=1)[0]
            self.concepts["priority"] = self.priority
            if sample == "normal":
                return
            else:
                self.concepts[sample] = 1.0
                return
        else:
            self.concepts["priority"] = self.priority
            return
    
    def get_action(self, local_action_dist):
        # Create a copy of the action distribution to modify
        valid_action_dist = local_action_dist.clone()

        # Iterate over all move actions to check for validity
        for action_index, move in self.action_to_move.items():
            next_pos = self.pos + move
            # Check if the next position is within the map boundaries and is a movable region
            if (next_pos[0] < 0 or next_pos[0] >= self.movable_region.shape[0] or
                next_pos[1] < 0 or next_pos[1] >= self.movable_region.shape[1] or
                not self.movable_region[next_pos[0], next_pos[1]]):
                    # If the move is invalid, set its probability to 0
                    valid_action_dist[action_index] = 0

        # If all actions have a probability of 0, stop
        if torch.sum(valid_action_dist) == 0:
            return self.action_space[-1]  # Return the stop action

        # Normalize the distribution and sample an action
        normalized_action_dist = valid_action_dist / valid_action_dist.sum()
        dist = Categorical(normalized_action_dist)
        action_index = dist.sample()
        return self.action_space[action_index]
    
    def get_available_actions(self):
        valid_action_dist = torch.ones(len(self.action_space), dtype=torch.bool, device=self.device)
        for action_index, move in self.action_to_move.items():
            next_pos = self.pos + move
            # Check if the next position is within the map boundaries and is a movable region
            if (next_pos[0] < 0 or next_pos[0] >= self.movable_region.shape[0] or
                next_pos[1] < 0 or next_pos[1] >= self.movable_region.shape[1] or
                not self.movable_region[next_pos[0], next_pos[1]]):
                    # If the move is invalid, set its probability to 0
                    valid_action_dist[action_index] = False
                    
        return valid_action_dist
        

    def get_next_action(self, world_state_matrix, local_action_dist):
        # reached goal
        if not self.reach_goal:
            if torch.all(self.pos == self.goal):
                if not self.debug:
                    self.reach_goal = True
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]