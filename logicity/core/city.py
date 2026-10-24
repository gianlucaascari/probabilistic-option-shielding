from .building import Building
from ..planners import LPlanner_mapper
import time
import numpy as np
import logging
import random
from skimage.draw import line, polygon
from scipy.ndimage import label
import torch
import torch.nn.functional as F
from ..core.config import *
from ..utils.vis import visualize_intersections
from ..utils.find import find_midroad_segments
from ..utils.gen import gen_occ

logger = logging.getLogger(__name__)

class City:
    def __init__(self, grid_size, local_planner, logic_engine_file=None, use_multi=False):
        self.grid_size = grid_size
        self.layers = BASIC_LAYER
        # 0 for blocks
        # 1 for buildings
        # 2 for streets
        # Initialize the grid with 0 placeholders
        # world_start_matrix
        self.city_grid = torch.zeros((self.layers, grid_size[0], grid_size[1]))
        self.buildings = []
        self.streets = []
        self.layer_id2agent_list_id = {}
        self.agents = []
        self.label2type = LABEL_MAP
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        # city rule defines local decision of all the agents
        self.local_planner = LPlanner_mapper[local_planner](logic_engine_file)
        self.use_multi = use_multi
        self.logic_grounds = {}
        # vis color map
        self.color_map = COLOR_MAP
        # optional military zone matrix (set by CityLoader for maps that define one)
        self.military_matrix = None

    def update(self):
        current_obs = {}
        # state at time t
        current_obs["World"] = self.city_grid.clone()
        current_obs["Agent_actions"] = []

        new_matrix = torch.zeros_like(self.city_grid)
        current_world = self.city_grid.clone()
        # first do local planning based on city rules, use the current world state, don't update the city matrix
        agent_action_dist = self.local_planner.plan(current_world, self.intersection_matrix, self.direction_matrix, self.agents, \
                                                    self.layer_id2agent_list_id, use_multiprocessing=self.use_multi)
        # Then do global action taking acording to the local planning results
        # get occupancy map
        for agent in self.agents:
            # re-initialized agents may update city matrix as well
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            empty_action = agent.action_dist.clone()
            local_action_dist = agent_action_dist[agent_name]
            # global trajectory-based action or sampling from local action distribution
            local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, local_action_dist)
            # save the current action in the action
            empty_action[local_action] = 1.0
            current_obs["Agent_actions"].append(empty_action)
            if agent.reach_goal:
                continue
            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[BASIC_LAYER:] = new_matrix[BASIC_LAYER:]
        return current_obs

    def add_building(self, building):
        """Add a building to the city and mark its position on the grid.

        If building.render is False the building type is NOT written to the
        BUILDING_ID layer, so the block still contributes its bounding box to
        add_mid / add_intersections but no icon is drawn by the renderer.
        """
        self.buildings.append(building)
        building_code = self.type2label[building.type]
        self.city_grid[0][building.position[0]:building.position[0] + building.size[0], \
            building.position[1]:building.position[1] + building.size[1]] = building.block
        if getattr(building, "render", True):
            self.city_grid[1][building.position[0]:building.position[0] + building.size[0], \
                building.position[1]:building.position[1] + building.size[1]] = building_code

    def add_street(self, street):
        """
        Add a street to the city and mark its position on the grid.
        
        Add direction matrix for intended traffic directions:
        - 0: up
        - 1: right
        - 2: down
        - 3: left
        """
        
        if not hasattr(self, 'direction_matrix'):
            self.direction_matrix = torch.zeros((4, self.grid_size[0], self.grid_size[1]), dtype=torch.bool)

        self.streets.append(street)
        street_code = self.type2label[street.type]
        
        if street.orientation == 'horizontal':
            y_start, x_start = street.position
            width, length = street.width, street.length
            y_end = y_start + width
            x_end = x_start + length
            
            # Update city grid for street
            street_area = self.city_grid[2, y_start:y_end, x_start:x_end]
            intersection_mask = (street_area != 0) & (street_area != street_code)
            street_area[:] = street_code
            street_area[intersection_mask] = INTERSECTION_CODE

            # Update direction matrix for bidirectional traffic
            # Split the horizontal street into top and bottom halves
            if street.type == "Traffic Street":
                mid_point = y_start + (width + 1) // 2
                # Top half: leftward traffic (layer 3)
                self.direction_matrix[3, y_start:(mid_point-1), x_start:x_end] = True
                # Bottom half: rightward traffic (layer 1)
                self.direction_matrix[1, mid_point:y_end, x_start:x_end] = True

        else:  # vertical street
            y_start, x_start = street.position
            length, width = street.length, street.width
            y_end = y_start + length
            x_end = x_start + width

            # Update city grid for street
            street_area = self.city_grid[2, y_start:y_end, x_start:x_end]
            intersection_mask = (street_area != 0) & (street_area != street_code)
            street_area[:] = street_code
            street_area[intersection_mask] = INTERSECTION_CODE

            # Update direction matrix for bidirectional traffic
            # Split the vertical street into left and right halves
            if street.type == "Traffic Street":
                mid_point = x_start + (width + 1) // 2
                # Left half: downward traffic (layer 2)
                self.direction_matrix[2, y_start:y_end, x_start:(mid_point-1)] = True
                # Right half: upward traffic (layer 0)
                self.direction_matrix[0, y_start:y_end, mid_point:x_end] = True

    def add_mid(self):
        """Add a mid lanes to traffic to the city and mark its position on the grid."""
        assert len(self.buildings) > 0
        street_code = self.type2label['Traffic Street'] + MID_LINE_CODE_PLUS
        
        # Get unique block IDs from the grid, excluding 0 (background)
        unique_blocks = torch.unique(self.city_grid[BLOCK_ID])
        unique_blocks = unique_blocks[unique_blocks != 0]
        
        for i in unique_blocks:
            i = i.item()
            current_block = self.city_grid[BLOCK_ID] == i
            pixels = torch.nonzero(current_block.float())
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            left = torch.min(cols).item()
            right = torch.max(cols).item()
            top = torch.min(rows).item()
            bottom = torch.max(rows).item()
            # top
            if top - TRAFFIC_STREET_WID >= 0:
                self.city_grid[STREET_ID][top-TRAFFIC_STREET_WID, left:(right+1)] = street_code
            if bottom + TRAFFIC_STREET_WID < self.grid_size[0]:
                self.city_grid[STREET_ID][bottom+TRAFFIC_STREET_WID, left:(right+1)] = street_code
            if left - TRAFFIC_STREET_WID >= 0:
                self.city_grid[STREET_ID][top:(bottom+1), left-TRAFFIC_STREET_WID] = street_code
            if right + TRAFFIC_STREET_WID < self.grid_size[1]:
                self.city_grid[STREET_ID][top:(bottom+1), right+TRAFFIC_STREET_WID] = street_code
        self.midline_matrix = (self.city_grid[STREET_ID] == street_code)
    
    def add_intersections(self):
        # Extract the 0-th layer of the world matrix
        world_layer = self.city_grid[BLOCK_ID, :, :]
        
        # Extract the unique block IDs from the 0-th layer
        unique_blocks = set(world_layer.flatten().tolist())
        if 0 in unique_blocks:
            unique_blocks.remove(0)  # Assuming 0 is the ID for non-block pixels
        
        if len(unique_blocks) == 0:
            self.intersection_matrix = torch.zeros((3, world_layer.shape[0], world_layer.shape[1]), dtype=bool)
            return

        if len(unique_blocks) == 1:
            self.intersection_matrix = torch.zeros((3, world_layer.shape[0], world_layer.shape[1]), dtype=bool)
            
            block_id = list(unique_blocks)[0]
            block_mask = (world_layer == block_id)
            rows, cols = torch.where(block_mask)
            top, bottom = rows.min().item()-1, rows.max().item()+1
            left, right = cols.min().item()-1, cols.max().item()+1
            
            w = TRAFFIC_STREET_WID
            
            # Define the 4 corners
            # Top-Left
            if top - w >= 0 and left - w >= 0:
                self.intersection_matrix[2, top-w:top, left-w:left] = True
            # Top-Right
            if top - w >= 0 and right + 1 + w <= self.grid_size[1]:
                self.intersection_matrix[2, top-w:top, right+1:right+1+w] = True
            # Bottom-Left
            if bottom + 1 + w <= self.grid_size[0] and left - w >= 0:
                self.intersection_matrix[2, bottom+1:bottom+1+w, left-w:left] = True
            # Bottom-Right
            if bottom + 1 + w <= self.grid_size[0] and right + 1 + w <= self.grid_size[1]:
                self.intersection_matrix[2, bottom+1:bottom+1+w, right+1:right+1+w] = True
            
            # Top-Left
            self.intersection_matrix[0, 3:4, 1:3] = True
            self.intersection_matrix[0, 0:2, 3:4] = True
            # Top-Right
            self.intersection_matrix[0, 3:4, 11:13] = True
            self.intersection_matrix[0, 1:3, 9:10] = True
            # Bottom-Left
            self.intersection_matrix[0, 9:10, 0:2] = True
            self.intersection_matrix[0, 10:12, 3:4] = True
            # Bottom-Right
            self.intersection_matrix[0, 11:13, 9:10] = True
            self.intersection_matrix[0, 9:10, 10:12] = True
            
            return
        
        # Find the corners of the blocks
        corners = {}
        minxmin, minymin = self.grid_size[0], self.grid_size[1]
        maxxmax, maxymax = 0, 0
        for block_id in unique_blocks:
            block_positions = (world_layer == block_id).nonzero()
            xmin, xmax = min(block_positions[:, 1])-1, max(block_positions[:, 1])+1
            ymin, ymax = min(block_positions[:, 0])-1, max(block_positions[:, 0])+1
            if xmin < minxmin:
                minxmin = xmin
            if ymin < minymin:
                minymin = ymin
            if xmax > maxxmax:
                maxxmax = xmax
            if ymax > maxymax:
                maxymax = ymax 
            corners[block_id] = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

        intersection_matrix = torch.zeros((3, world_layer.shape[0], world_layer.shape[1]), dtype=bool)
        intersection_line_len = TRAFFIC_STREET_WID + 2*WALKING_STREET_WID - 1
        # intersection line will also be determined by the mid line
        midroad_segments = find_midroad_segments(self.midline_matrix)
        end_lists = []
        mid_lists = []
        for segment in midroad_segments:
            mid_start, mid_end = segment
            if mid_start[0] == mid_end[0]:
                assert mid_end[1] > mid_start[1]
                # horizonal mid line
                bottom_e = mid_end + torch.tensor([1, 1])
                mid_lists.append(mid_end + torch.tensor([0, 1]))
                end_lists.append(bottom_e)
                top_e = mid_start + torch.tensor([-1, -1])
                end_lists.append(top_e)
                mid_lists.append(mid_start + torch.tensor([0, -1]))
            elif mid_start[1] == mid_end[1]:
                assert mid_end[0] > mid_start[0]
                # vertical mid line
                left_e = mid_end + torch.tensor([1, -1])
                mid_lists.append(mid_end + torch.tensor([1, 0]))
                end_lists.append(left_e)
                right_e = mid_start + torch.tensor([-1, 1])
                mid_lists.append(mid_start + torch.tensor([-1, 0]))
                end_lists.append(right_e)
                
        end_lists = torch.stack(end_lists, dim=0)
        mid_lists = torch.stack(mid_lists, dim=0)
        mid_x, mid_y = mid_lists.t()
        end_x, end_y = end_lists.t()
        for block_id, block_corners in corners.items():
            for other_block_id, other_block_corners in corners.items():
                if block_id != other_block_id:
                    for corner in block_corners:
                        for other_corner in other_block_corners:
                            corner_dis = np.linalg.norm(np.array(corner) - np.array(other_corner))
                            if corner_dis == intersection_line_len:
                                local_line = torch.zeros_like(intersection_matrix[0])
                                rr, cc = line(corner[0], corner[1], other_corner[0], other_corner[1])
                                # Third layer is for check "In intersection, they are blocks", here is for T-junctions
                                if rr.max()==rr.min():
                                    if rr.min() == minxmin:
                                        # T junction
                                        intersection_matrix[2, 0:rr.min()+1, cc.min():cc.max()+1] = True
                                    elif rr.max() == maxxmax:
                                        # T junction
                                        intersection_matrix[2, rr.max():, cc.min():cc.max()+1] = True
                                else:
                                    if cc.min() == minymin:
                                        # T junction
                                        intersection_matrix[2, rr.min():rr.max()+1, 0:cc.min()+1] = True
                                    elif cc.max() == maxymax:
                                        # T junction
                                        intersection_matrix[2, rr.min():rr.max()+1, cc.max():] = True
                                # First and second layer is for check "At intersection, they are lines that about to **enter** an intersection"
                                # Second layer checks pedestrains, they are allowed to enter the intersection in both directions
                                # We use the mid line and end points to determine the entrance line
                            elif corner_dis > 1.4*intersection_line_len and corner_dis < 1.5*intersection_line_len:
                                # second layer is for check "In intersection, they are blocks"
                                rr, cc = line(corner[0], corner[1], other_corner[0], other_corner[1])
                                # Gather the vertices of the polygon
                                assert rr.max()!=rr.min() and cc.max()!=cc.min()
                                intersection_matrix[2, rr.min():rr.max()+1, cc.min():cc.max()+1] = True
                                
        # Label connected regions in the intersection matrix
        # Check if the number of connected regions is correct, car lines, ped lines, and blocks
        # Assertions only apply to the original WORLD_SIZE grid; larger maps relax them.
        _is_standard_grid = tuple(self.grid_size) == (WORLD_SIZE, WORLD_SIZE)
        _, num_line = label(intersection_matrix[0])
        if _is_standard_grid:
            assert num_line == NUM_INTERSECTIONS_LINES, "Number of intersection lines for cars is not {}".format(NUM_INTERSECTIONS_LINES)
        _, num_line = label(intersection_matrix[1])
        labeled_matrix_block, num_block = label(intersection_matrix[2])
        if _is_standard_grid:
            assert num_block == NUM_INTERSECTIONS_BLOCKS, "Number of intersection blocks is not {}".format(NUM_INTERSECTIONS_BLOCKS)
            assert num_line == NUM_INTERSECTIONS_BLOCKS, "Number of intersection lines for peds is not {}".format(NUM_INTERSECTIONS_BLOCKS)
        # Label the intersection matrix, they share the same ID
        labeled_matrix_line = intersection_matrix[0].numpy().astype(labeled_matrix_block.dtype) * labeled_matrix_block
        labeled_matrix_line_ped = intersection_matrix[1].numpy().astype(labeled_matrix_block.dtype) * labeled_matrix_block
        # Important: Exclude the intersection lines in the blocks, so that the agent AT intersetion is not counted as IN the intersection
        labeled_matrix_block[labeled_matrix_line_ped!=0] = 0
        intersection_matrix = np.array([labeled_matrix_line, labeled_matrix_line_ped, labeled_matrix_block])
        self.intersection_matrix = torch.tensor(intersection_matrix)

    def add_agent(self, agent):
        """Add a agents to the city and mark its position on the grid. Label correspons
            to the current position, Label+0.1 denotes planned global path, Label+0.3 denotes goal point,
            Label+0.2 denoets next position and, Label-0.1 means walked position, Label-0.2 means start position,
            see config.py for details.
        """
        self.agents.append(agent)
        agent_layer = torch.zeros((1, self.grid_size[0], self.grid_size[1]))
        agent_code = self.type2label[agent.type]
        # draw agent
        agent_layer[0][agent.goal[0], agent.goal[1]] = agent_code + AGENT_GOAL_PLUS
        agent_layer[0][agent.start[0], agent.start[1]] = agent_code
        if hasattr(agent, "global_traj"):
            for way_points in agent.global_traj[1:-1]:
                if torch.all(way_points==agent.start) or torch.all(way_points==agent.goal):
                    continue
                agent_layer[0][way_points[0], way_points[1]] = agent_code + AGENT_GLOBAL_PATH_PLUS
        agent.layer_id = self.city_grid.shape[0]
        self.layer_id2agent_list_id[agent.layer_id] = len(self.agents)-1
        self.city_grid = torch.concat([self.city_grid, agent_layer], dim=0)
        self.layers += 1
        agent_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        if "{}_{}".format(agent.type, agent.id) not in self.color_map.keys():
                self.color_map["{}_{}".format(agent.type, agent.id)] = agent_color
                
    def remove_agent(self, agent_id):
        target_idx = None
        for idx, agent in enumerate(self.agents):
            if agent.id == agent_id:
                target_idx = idx
                break

        if target_idx is None:
            return None

        agent = self.agents.pop(target_idx)
        layer_id = agent.layer_id

        if BASIC_LAYER <= layer_id < self.city_grid.shape[0]:
            self.city_grid = torch.cat(
                [self.city_grid[:layer_id], self.city_grid[layer_id + 1 :]], dim=0
            )
            self.layers -= 1

        self.layer_id2agent_list_id.clear()
        for idx, existing_agent in enumerate(self.agents):
            if existing_agent.layer_id > layer_id:
                existing_agent.layer_id -= 1
            self.layer_id2agent_list_id[existing_agent.layer_id] = idx

        self.color_map.pop(f"{agent.type}_{agent.id}", None)
        return agent
    
    def get_agent(self, agent_id):
        target_idx = None
        for idx, agent in enumerate(self.agents):
            if agent.id == agent_id:
                target_idx = idx
                break

        if target_idx is None:
            return None

        return self.agents[target_idx]