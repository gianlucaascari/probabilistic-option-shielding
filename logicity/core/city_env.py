import random

import numpy as np
import torch

from .city import City
from .config import *
from ..utils.pred_converter.z3 import OPPOSITE_DIRECTION_MATRIX


class CityEnv(City):
    """City with one agent controlled by an RL policy.

    ``rl_agent["random_pedestrian"]`` defines the school-time pedestrians: at
    every episode start school time is drawn with ``school_time_probability``,
    and the pedestrians are placed with ``ped_prob_in_school_time`` during
    school time and ``ped_prob_out_school_time`` otherwise.
    """

    def __init__(self, grid_size, local_planner, logic_engine_file, rl_agent, use_multi=False, device="cpu"):
        super().__init__(grid_size, local_planner, logic_engine_file, use_multi=use_multi)
        self.rl_agent = rl_agent
        self.use_logic_obs = rl_agent.get("use_logic_obs", True)
        self.logic_grounding_shape, self.pred_grounding_index = self.local_planner.logic_grounding_shape(self.rl_agent["fov_entities"])
        if not self.use_logic_obs:
            self.logic_grounding_shape = 3  # override: just pos_x, pos_y, school_time
        self.device = device
        self.school_time = False
        if "random_pedestrian" in rl_agent:
            rp = rl_agent["random_pedestrian"]
            self.school_ped_ids = rp["ped_ids"]
            self.school_time_probability = rp["school_time_probability"]
            self.ped_prob_in_school_time = rp["ped_prob_in_school_time"]
            self.ped_prob_out_school_time = rp["ped_prob_out_school_time"]

    def store_school_pedestrians(self):
        if "random_pedestrian" in self.rl_agent:
            self.school_peds = [self.get_agent(pid) for pid in self.school_ped_ids]

    def random_pedestrian(self):
        """Draw school time and place the school pedestrians accordingly."""
        if "random_pedestrian" in self.rl_agent:
            for pid in self.school_ped_ids:
                self.remove_agent(pid)
            self.school_time = random.random() < self.school_time_probability
            if self.school_time and random.random() < self.ped_prob_in_school_time:
                for ped in self.school_peds:
                    self.add_agent(ped)
            if not self.school_time and random.random() < self.ped_prob_out_school_time:
                for ped in self.school_peds:
                    self.add_agent(ped)

    def move_rl_agent(self, action, idx):
        current_obs = {}
        current_obs["Fail"] = []
        current_obs["Reward"] = []
        current_obs["Violations"] = []
        
        fail, reward, violations = self.local_planner.eval(action)
        current_obs["Fail"].append(fail)
        current_obs["Reward"].append(reward)
        current_obs["Violations"].extend(violations)
        new_matrix = torch.zeros_like(self.city_grid)
        
        for agent in self.agents:
            # re-initialized agents may update city matrix as well
            # local reasoning-based action distribution
            # global trajectory-based action or sampling from local action distribution
            if agent.layer_id == idx:
                local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, action)
            else: 
                continue

            if agent.reach_goal:
                continue

            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            
            if torch.all(agent.pos == agent.goal):
                current_obs["Reward"][-1] += self.rl_agent["reward_for_goal"]

            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[idx] = new_matrix[idx]
        
        return current_obs
    
    def get_rl_agent_valid_actions(self, layer_id):
        for agent in self.agents:
            if agent.layer_id == layer_id:
                return agent.get_available_actions()
            
    def get_sensor_values(self, layer_id):
        sensors = self.rl_agent["sensors"]
        sensor_number = len(sensors) * 4
        
        sensor_values = torch.zeros(sensor_number, dtype=torch.float32, device=self.device) + 0.00001
        
        for a in self.agents:
            if a.layer_id == layer_id:
                agent = a
        
        pos = {}
        pos["left"] = agent.pos + torch.tensor((0, -1))
        pos["right"] = agent.pos + torch.tensor((0, 1))
        pos["up"] = agent.pos + torch.tensor((-1, 0))
        pos["down"] = agent.pos + torch.tensor((1, 0))
        
        idx = 0
        
        if "no_drive_area" in sensors:
            for i, direction in enumerate(['left', 'right', 'up', 'down']):
                if pos[direction][0] >= agent.movable_region.shape[0] or pos[direction][1] >= agent.movable_region.shape[1] or pos[direction][0] <= -1 or pos[direction][1] <= -1:
                    sensor_values[i] = 1.0
                else:
                    sensor_values[i] = not agent.movable_region[pos[direction][0], pos[direction][1]]
            idx += 4
        
        if "against_traffic" in sensors:
            for i, direction in enumerate(['left', 'right', 'up', 'down']):
                allowed_directions = self.direction_matrix[:, agent.pos[0], agent.pos[1]]
                
                for j, d in enumerate(allowed_directions):
                    sensor_values[i + idx] = 1.0
                    if d and not OPPOSITE_DIRECTION_MATRIX[direction.capitalize()][j]:
                        sensor_values[i + idx] = 0.0
                        break
            idx += 4
                
        if "other_agent" in sensors:
            for j in range(3, self.city_grid.shape[0]):
                if j != layer_id:
                    for i, direction in enumerate(['left', 'right', 'up', 'down']):
                        if pos[direction][0] >= agent.movable_region.shape[0] or pos[direction][1] >= agent.movable_region.shape[1] or pos[direction][0] <= -1 or pos[direction][1] <= -1:
                            sensor_values[idx + i] = 0
                        else:
                            val = self.city_grid[j][pos[direction][0]][pos[direction][1]]
                            if val != 0 and (val % 1.0) == 0.0:
                                sensor_values[idx + i] = 1
            idx += 4

        if "military_area" in sensors:
            mat = getattr(self, 'military_matrix', None)
            for i, direction in enumerate(['left', 'right', 'up', 'down']):
                p = pos[direction]
                if (p[0] < 0 or p[1] < 0 or
                        p[0] >= self.city_grid.shape[1] or p[1] >= self.city_grid.shape[2]):
                    sensor_values[idx + i] = 0.0
                else:
                    sensor_values[idx + i] = float(mat[p[0], p[1]]) if mat is not None else 0.0
            idx += 4

        return sensor_values.unsqueeze(0)

    def update(self, idx):
        current_obs = {}
        # state at time t
        current_obs["World_state"] = []
        current_obs["Ground_dic"] = []

        new_matrix = torch.zeros_like(self.city_grid)
        current_world = self.city_grid.clone()
        # first do local planning based on city rules
        agent_action_dist = self.local_planner.plan(current_world, self.intersection_matrix, self.direction_matrix, self.agents, \
                                                    self.layer_id2agent_list_id, use_multiprocessing=self.use_multi, rl_agent=idx)
        # Then do global action taking acording to the local planning results
        
        for i, agent in enumerate(self.agents):
            # re-initialized agents may update city matrix as well
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            # local reasoning-based action distribution
            # global trajectory-based action or sampling from local action distribution
            if agent.layer_id == idx: 
                pos_x = agent.pos[0].item()
                pos_y = agent.pos[1].item()
                
                if self.use_logic_obs:
                    world_state = np.append(agent_action_dist["{}_grounding".format(agent_name)], [pos_x, pos_y, self.school_time])
                else:
                    world_state = np.array([pos_x, pos_y, self.school_time], dtype=np.float32)
                current_obs["World_state"].append(world_state)
                new_matrix[agent.layer_id] = self.city_grid[agent.layer_id].clone()
                if "{}_grounding_dic".format(agent_name) in agent_action_dist:
                    current_obs["Ground_dic"].append(agent_action_dist["{}_grounding_dic".format(agent_name)])
                    current_obs['Ground_dic'][i]['pos_x'] = pos_x
                    current_obs['Ground_dic'][i]['pos_y'] = pos_y
                    current_obs['Ground_dic'][i]['school_t'] = self.school_time
                continue
            else: 
                local_action_dist = agent_action_dist[agent_name]
                local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, local_action_dist)

            if agent.reach_goal:
                continue

            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[BASIC_LAYER:] = new_matrix[BASIC_LAYER:]
        return current_obs
