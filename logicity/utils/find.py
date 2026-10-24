import torch
import torch.nn.functional as F
from ..core.config import *

def find_nearest_building(world_state_matrix, start_point):
    start_point = start_point.clone().detach()
    buildings_layer = world_state_matrix[BLOCK_ID].clone().detach()
    
    building_positions = torch.nonzero(buildings_layer)
    distances = torch.abs(building_positions - start_point).sum(dim=1)
    
    min_index = distances.argmin().item()
    nearest_building = building_positions[min_index]
    
    return nearest_building

def find_building_mask(city_grid, start_point):
    # The value at the starting point, i.e., the building type
    start_point = start_point.numpy().tolist()
    city_grid = city_grid[BLOCK_ID]
    building_type = city_grid[start_point[0], start_point[1]]
    
    # Expand boundaries iteratively to find the extent of the building
    top, bottom, left, right = start_point[0], start_point[0], start_point[1], start_point[1]

    while top > 0 and city_grid[top - 1, start_point[1]] == building_type:
        top -= 1

    while bottom < city_grid.shape[0] - 1 and city_grid[bottom + 1, start_point[1]] == building_type:
        bottom += 1

    while left > 0 and city_grid[start_point[0], left - 1] == building_type:
        left -= 1

    while right < city_grid.shape[1] - 1 and city_grid[start_point[0], right + 1] == building_type:
        right += 1

    # Create the mask
    mask = torch.zeros_like(city_grid, dtype=torch.bool)
    mask[top:bottom + 1, left:right + 1] = True

    return mask

def find_midroad_segments(midline_matrix):
    midroad_segments = []

    # Scan horizontally for road segments
    for row in range(midline_matrix.shape[0]):
        start_col = None
        for col in range(midline_matrix.shape[1]):
            if midline_matrix[row][col] == 1:
                if start_col is None:
                    start_col = col
            else:
                if start_col is not None:
                    if col - 1 - start_col > 0:
                        midroad_segments.append((torch.tensor([row, start_col]), torch.tensor([row, col - 1])))
                    start_col = None

    # Scan vertically for road segments
    for col in range(midline_matrix.shape[1]):
        start_row = None
        for row in range(midline_matrix.shape[0]):
            if midline_matrix[row][col] == 1:
                if start_row is None:
                    start_row = row
            else:
                if start_row is not None:
                    if row - 1 - start_row > 0:
                        midroad_segments.append((torch.tensor([start_row, col]), torch.tensor([row - 1, col])))
                    start_row = None

    return midroad_segments

def interpolate_car_path(movable_map, path_on_graph, max_step):
    interpolated = []
    interpolated.append(torch.tensor(path_on_graph[0]))
    for i in range(len(path_on_graph) - 1):
        current_point = path_on_graph[i]
        next_point = path_on_graph[i+1]

        while current_point != next_point:

            # Determine the difference in X and Y
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]

            if dx != 0 and dy != 0:
                # Move to the intersection if both dx and dy are non-zero
                assert abs(dx) == abs(dy)
                if movable_map[current_point[0], next_point[1]]:
                    intersect = (current_point[0], next_point[1])
                else:
                    movable_map[next_point[0], current_point[1]]
                    intersect = (next_point[0], current_point[1])
                # First move to intersect:
                while current_point != intersect:
                    dx = intersect[0] - current_point[0]
                    dy = intersect[1] - current_point[1]
                    if abs(dx) >= max_step:
                        step = max_step * int(dx/abs(dx))
                        current_point = (current_point[0] + step, current_point[1])
                    elif abs(dy) >= max_step:
                        step = max_step * int(dy/abs(dy))
                        current_point = (current_point[0], current_point[1] + step)
                    else:
                        if dx != 0:
                            step = int(dx)
                            current_point = (current_point[0] + step, current_point[1])
                        elif dy != 0:
                            step = int(dy)
                            current_point = (current_point[0], current_point[1] + step)
                    interpolated.append(torch.tensor(current_point))
            else:
                # If the vehicle doesn't need to turn, just move straight
                if abs(dx) >= max_step:
                    step = max_step * int(dx/abs(dx))
                    current_point = (current_point[0] + step, current_point[1])
                elif abs(dy) >= max_step:
                    step = max_step * int(dy/abs(dy))
                    current_point = (current_point[0], current_point[1] + step)
                else:
                    if dx != 0:
                        step = int(dx)
                        current_point = (current_point[0] + step, current_point[1])
                    elif dy != 0:
                        step = int(dy)
                        current_point = (current_point[0], current_point[1] + step)
                interpolated.append(torch.tensor(current_point))

    return torch.stack(interpolated, dim=0)

def find_agent(agents, entity_name):
    _, agent_type, agent_id = entity_name.split("_")
    for agent in agents:
        if agent.type == agent_type and agent.layer_id == int(agent_id):
            return agent
    return None

def find_entity(agent):
    agent_type = agent.type
    # This is layer id, not agent id
    agent_id = agent.layer_id
    agent_name = "Agents_{}_{}".format(agent_type, agent_id)
    return agent_name