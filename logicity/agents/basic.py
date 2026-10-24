import torch
from torch.distributions import Categorical
from ..core.config import *

class Agent:
    def __init__(self, size, id, world_state_matrix, concepts, init_info=None, debug=False, region=240):
        self.size = size
        self.concepts = concepts
        self.type = concepts["type"]
        self.priority = concepts["priority"]
        self.id = id
        # -1 is always the stop action
        # Actions: ["left", "right", "up", "down", "stop"]
        self.action_space = torch.tensor(range(5))
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_1", 
            1: "Right_1", 
            2: "Up_1", 
            3: "Down_1", 
            4: "Stop"
            }
        # move direction will be used to determine FOV and Spatial Predicates
        self.last_move_dir = None
        self.start = None
        self.goal = None
        self.pos = None
        self.layer_id = 0
        self.reach_goal = False
        self.reach_goal_buffer = 0
        self.debug = debug
        self.region = region
        self.init(world_state_matrix, init_info, debug)

    def init(self, world_state_matrix, init_info=None, debug=False):
        # init global planner, global traj, local planner, start and goal point
        pass

    def get_start(self, world_state_matrix):
        pass

    def get_goal(self, world_state_matrix):
        pass
        
    def move(self, action, ped_layer, curr_label):
        pass
        
    def get_next_action(self, world_state_matrix):
        pass

    def get_movable_area(self, world_state_matrix):
        pass

    def get_action(self, local_action_dist):
        if len(local_action_dist.nonzero()) == 1:
            # local planner is very strict, only one action is possible
            final_action_dist = local_action_dist
        else:
            global_action = self.get_global_action()
            if len(global_action.nonzero()) == 1:
            # local planner gives multiple actions, but global planner is very strict
                final_action_dist = global_action
            else:
                # local planner gives multiple actions, use global planner to filter
                final_action_dist = torch.logical_and(local_action_dist, global_action).float()
                if len(final_action_dist.nonzero()) == 0:
                    # local planner and global planner has conflict, but local planner is not strict, take global
                    final_action_dist = global_action
        # now only one action is possible
        assert len(final_action_dist.nonzero()) >= 1
        # sample from the local planner
        normalized_action_dist = final_action_dist / final_action_dist.sum()
        dist = Categorical(normalized_action_dist)
        # Sample an action index from the distribution
        action_index = dist.sample()
        # Get the actual action from the action space using the sampled index
        return self.action_space[action_index]

    def move(self, action, ped_layer):
        curr_pos = torch.nonzero((ped_layer==TYPE_MAP[self.type]).float())[0]
        assert torch.all(self.pos == curr_pos), (self.pos, curr_pos)
        next_pos = self.pos.clone()
        # becomes walked grid
        ped_layer[self.pos[0], self.pos[1]] += AGENT_WALKED_PATH_PLUS
        next_pos += self.action_to_move.get(action.item(), torch.tensor((0, 0)))
        self.pos = next_pos.clone()
        # Update Agent Map
        ped_layer[self.start[0], self.start[1]] = TYPE_MAP[self.type] + AGENT_START_PLUS
        ped_layer[self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]
        # Update last move direction
        move_dir = self.action_mapping[action.item()].split("_")[0]
        if move_dir != "Stop":
            self.last_move_dir = move_dir
        return ped_layer

    def get_global_action(self):
        global_action_dist = torch.zeros_like(self.action_space).float()
        current_pos = torch.all((self.global_traj == self.pos), dim=1).nonzero()[0]
        next_pos = current_pos + 1 if current_pos < len(self.global_traj) - 1 else 0
        del_pos = self.global_traj[next_pos] - self.pos
        if del_pos.shape == torch.Size([2]):        # workaround to allow still pedestrian
            global_action_dist[-1] = 1.0
            return global_action_dist
        for move in self.move_to_action.keys():
            if torch.dot(del_pos.squeeze(), move) > 0:
                next_point = self.pos + move
                step = torch.max(torch.abs(move)).item()
                # 1. if the next point is on the global traj
                if len(torch.all((self.global_traj[next_pos:next_pos+step] == next_point), dim=1).nonzero()) > 0:
                    global_action_dist[self.move_to_action[move]] = 1.0
        if torch.all(global_action_dist==0):
            global_action_dist[-1] = 1.0
            if torch.all(del_pos==0):
                self.global_traj.pop(next_pos)
        return global_action_dist
    
    def init_from_dict(self, init_info):
        self.start = torch.tensor(init_info["start"])
        self.pos = self.start.clone()
        self.goal = torch.tensor(init_info["goal"])
        self.type = init_info["type"]
        self.priority = init_info["priority"]
        self.concepts = init_info["concepts"]